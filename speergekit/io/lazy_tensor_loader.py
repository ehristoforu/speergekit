# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import json
import logging
import os
import os.path
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import safetensors
import safetensors.torch
import torch
from torch import Tensor

from speergekit.io.loader import TensorLoader


@dataclass
class ShardInfo:
    """
    Information about a single shard in a sharded model.
    """
    filename: str
    contained_keys: List[str]


@dataclass
class ShardedTensorIndex:
    """
    An index of tensors in a sharded model.
    """
    base_path: str
    is_safetensors: bool
    tensor_paths: Dict[str, str]
    shards: List[ShardInfo]

    @classmethod
    def from_disk(cls, base_path: str) -> "ShardedTensorIndex":
        """
        Creates a ShardedTensorIndex from a model on disk.
        """
        model_path = None
        for model_file_name in [
            "model.safetensors",
            "pytorch_model.bin",
        ]:
            candidate_path = os.path.join(base_path, model_file_name)
            if os.path.exists(candidate_path) or os.path.exists(
                candidate_path + ".index.json"
            ):
                model_path = candidate_path
                break

        if not model_path:
            raise RuntimeError(f"Unable to find model files at {base_path}")

        is_safetensors = model_path.endswith(".safetensors")
        tensor_paths = None
        shards = []

        if os.path.exists(model_path + ".index.json"):
            # shared model - parse index
            with open(model_path + ".index.json", "r", encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
            tensor_paths = weight_map
            shard_names = sorted(list(set(weight_map.values())))
            for shard_name in shard_names:
                shards.append(
                    ShardInfo(
                        shard_name,
                        [
                            key
                            for key, value in tensor_paths.items()
                            if value == shard_name
                        ],
                    )
                )

            return ShardedTensorIndex(
                base_path=base_path,
                is_safetensors=is_safetensors,
                tensor_paths=tensor_paths,
                shards=shards,
            )

        elif os.path.exists(model_path):
            return ShardedTensorIndex.from_file(model_path)

        else:
            raise RuntimeError(f"Unable to find model files at {base_path}")

    @classmethod
    def from_file(cls, file_path: str) -> "ShardedTensorIndex":
        """
        Creates a ShardedTensorIndex from a single file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        lower = file_path.lower()
        shard_name = os.path.basename(file_path)
        if lower.endswith(".safetensors"):
            with safetensors.safe_open(file_path, framework="pt") as st:
                tensor_paths = {key: shard_name for key in st.keys()}
        else:
            shard = torch.load(file_path, map_location="meta")
            if "state_dict" in shard:
                shard = shard["state_dict"]

            tensor_paths = {key: shard_name for key in shard}

        return ShardedTensorIndex(
            base_path=os.path.dirname(file_path),
            is_safetensors=lower.endswith(".safetensors"),
            tensor_paths=tensor_paths,
            shards=[ShardInfo(shard_name, list(tensor_paths.keys()))],
        )


class LazyTensorLoader:
    """
    A tensor loader that lazily loads tensors from a sharded model.
    """
    index: ShardedTensorIndex
    current_shard: Optional[TensorLoader]
    lazy_unpickle: bool
    lock: threading.Lock

    def __init__(self, index: ShardedTensorIndex, lazy_unpickle: bool = True):
        """
        Initializes a new instance of the LazyTensorLoader class.

        Args:
            index: The index of the sharded model.
            lazy_unpickle: Whether to use lazy unpickling.
        """
        self.index = index
        self.current_shard = None
        self.lazy_unpickle = lazy_unpickle
        self.lock = threading.Lock()

    def get_tensor(
        self,
        key: str,
        device: str = "cpu",
        aliases: Optional[List[str]] = None,
        raise_on_missing: bool = True,
    ) -> Optional[Tensor]:
        """
        Gets a tensor from the sharded model.

        Args:
            key: The key of the tensor to get.
            device: The device to load the tensor on.
            aliases: A list of aliases for the tensor key.
            raise_on_missing: Whether to raise an error if the tensor is not found.

        Returns:
            The tensor, or None if the tensor is not found and raise_on_missing is False.
        """
        if aliases and key not in self.index.tensor_paths:
            for alias in aliases:
                if alias in self.index.tensor_paths:
                    key = alias
                    break

        with self.lock:
            if self.current_shard is None or key not in self.current_shard.keys():
                if key not in self.index.tensor_paths:
                    if raise_on_missing:
                        raise KeyError(key)
                    return None

                self.current_shard = None
                self.current_keys = None

                shard_file = self.index.tensor_paths[key]
                shard_full_path = os.path.join(self.index.base_path, shard_file)
                logging.debug(f"Opening shard {shard_full_path}")
                self.current_shard = TensorLoader.get(
                    shard_full_path, use_lazy_unpickle=self.lazy_unpickle, device=device
                )

            return self.current_shard.get_tensor(key).to(device)

    def flush(self):
        """
        Flushes the current shard, releasing any resources it holds.
        """
        with self.lock:
            self.current_shard = None
            self.current_keys = None

    @classmethod
    def from_disk(
        cls, base_path: str, lazy_unpickle: bool = True
    ) -> "LazyTensorLoader":
        """
        Creates a LazyTensorLoader from a model on disk.

        Args:
            base_path: The path to the model.
            lazy_unpickle: Whether to use lazy unpickling.

        Returns:
            A new instance of the LazyTensorLoader class.
        """
        return LazyTensorLoader(ShardedTensorIndex.from_disk(base_path), lazy_unpickle)