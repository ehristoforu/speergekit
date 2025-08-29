from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from speergekit.io.lazy_tensor_loader import LazyTensorLoader
from speergekit.merge_methods.base import BaseMergeMethod


class LinearMerge(BaseMergeMethod):
    """
    A merge method that performs a linear combination of models.
    """

    def run(
        self,
        parameters: Dict[str, Any],
        models: List[LazyTensorLoader],
        base_model: Optional[LazyTensorLoader] = None,
    ) -> Dict[str, torch.Tensor]:
        if base_model:
            raise ValueError("Linear merge does not support a base model.")

        if len(models) > 6:
            raise ValueError("Linear merge supports up to 6 models.")

        # Collect all tensor keys from all models
        tensor_keys = set()
        for model in models:
            tensor_keys.update(model.index.tensor_paths.keys())

        merged_state_dict = {}
        if "weights" in parameters:
            weights = parameters["weights"]
        else:
            weights = [1.0 / len(models)] * len(models)

        for key in tqdm(tensor_keys, desc="Merging tensors"):
            tensors = []
            present_weights = []
            for i, model in enumerate(models):
                if key in model.index.tensor_paths:
                    tensors.append(model.get_tensor(key))
                    present_weights.append(weights[i])

            if not tensors:
                continue

            # Check that all tensors have the same shape
            first_shape = tensors[0].shape
            for i, t in enumerate(tensors):
                if t.shape != first_shape:
                    raise ValueError(
                        f"Tensor shape mismatch for key {key} in model {i}. Expected {first_shape}, got {t.shape}."
                    )

            stacked_tensors = torch.stack(tensors, dim=0)

            tensor_weights = torch.tensor(
                present_weights, dtype=stacked_tensors.dtype, device=stacked_tensors.device
            )
            while len(tensor_weights.shape) < len(stacked_tensors.shape):
                tensor_weights.unsqueeze_(-1)

            merged_tensor = (tensor_weights * stacked_tensors).sum(dim=0)
            if parameters.get("normalize", True):
                merged_tensor = merged_tensor / tensor_weights.sum()

            merged_state_dict[key] = merged_tensor

        return merged_state_dict