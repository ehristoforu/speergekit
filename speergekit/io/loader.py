# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from abc import ABC, abstractmethod
import os
from typing import Dict, Optional, Sequence

import safetensors
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

from speergekit.io.lazy_unpickle import (
    DeferredLoad,
    LazyUnpickleModule,
    TorchArchiveReader,
    torch_lazy_load,
)


def load_model(
    model_path: str, use_lazy_unpickle: bool = False, device: Optional[str] = None
) -> "TensorLoader":
    """
    Loads a model from a local path or downloads it from the HuggingFace Hub.
    This handles a single model file, not a sharded model directory.
    """
    if not os.path.exists(model_path):
        try:
            # Try to download safetensors first
            model_path = hf_hub_download(repo_id=model_path, filename="model.safetensors")
        except (HfHubHTTPError, EntryNotFoundError):
            try:
                # If safetensors fails, try pytorch_model.bin
                model_path = hf_hub_download(
                    repo_id=model_path, filename="pytorch_model.bin"
                )
            except (HfHubHTTPError, EntryNotFoundError) as e:
                raise RuntimeError(
                    f"Could not find model.safetensors or pytorch_model.bin in repo {model_path}"
                ) from e

    return TensorLoader.get(
        model_path, use_lazy_unpickle=use_lazy_unpickle, device=device
    )


class TensorLoader(ABC):
    """Base class for (potentially lazy) tensor loaders."""

    @abstractmethod
    def get_tensor(self, key: str) -> torch.Tensor: ...

    @abstractmethod
    def keys(self) -> Sequence[str]: ...

    @classmethod
    def get(
        cls,
        shard_path: str,
        use_lazy_unpickle: bool = False,
        device: Optional[str] = None,
    ) -> "TensorLoader":
        if shard_path.lower().endswith(".safetensors"):
            # not a subclass of TensorLoader, but exposes same api
            return safetensors.safe_open(
                shard_path, framework="pt", device=device or "cpu"
            )
        elif use_lazy_unpickle:
            return LazyPickleLoader(shard_path, device=device)
        return DumbPytorchLoader(shard_path, device=device)


class LazyPickleLoader(TensorLoader):
    """Loader for pytorch files using a custom unpickler and vigorous monkeypatching."""

    zip_reader: TorchArchiveReader
    index: Dict[str, DeferredLoad]
    device: Optional[str] = None

    def __init__(self, path: str, device: Optional[str] = None):
        self.zip_reader = TorchArchiveReader(path)
        self.device = device
        with torch_lazy_load():
            self.index = torch.load(path, pickle_module=LazyUnpickleModule)

    def get_tensor(self, key: str) -> torch.Tensor:
        if key not in self.index:
            raise KeyError(key)

        return self.index[key].execute(self.zip_reader, map_location=self.device)

    def keys(self) -> Sequence[str]:
        return self.index.keys()


class DumbPytorchLoader(TensorLoader):
    """Naive `torch.load` shard loading."""

    tensors: Dict[str, torch.Tensor]

    def __init__(self, path: str, device: Optional[str] = None):
        self.tensors = torch.load(path, map_location=device, weights_only=True)

    def get_tensor(self, key: str) -> torch.Tensor:
        return self.tensors[key]

    def keys(self) -> Sequence[str]:
        return self.tensors.keys()