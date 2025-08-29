from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from speergekit.io.lazy_tensor_loader import LazyTensorLoader
from speergekit.merge_methods.base import BaseMergeMethod


class TaskArithmeticMerge(BaseMergeMethod):
    """
    A merge method that performs task arithmetic.
    """

    def run(
        self,
        parameters: Dict[str, Any],
        models: List[LazyTensorLoader],
        base_model: Optional[LazyTensorLoader] = None,
    ) -> Dict[str, torch.Tensor]:
        if not base_model:
            raise ValueError("Task arithmetic merge requires a base model.")
        if len(models) != 2:
            raise ValueError("Task arithmetic merge requires exactly two models.")

        scaling_factor = parameters.get("scaling_factor", 1.0)
        model_1, model_2 = models[0], models[1]

        # Collect all tensor keys
        tensor_keys = set(model_1.index.tensor_paths.keys())
        tensor_keys.update(model_2.index.tensor_paths.keys())
        tensor_keys.update(base_model.index.tensor_paths.keys())

        merged_state_dict = {}

        for key in tqdm(tensor_keys, desc="Merging tensors"):
            base_tensor = base_model.get_tensor(key)
            tensor_1 = model_1.get_tensor(key)
            tensor_2 = model_2.get_tensor(key)

            if tensor_1 is None or tensor_2 is None or base_tensor is None:
                continue

            if not (tensor_1.shape == tensor_2.shape == base_tensor.shape):
                raise ValueError(f"Tensor shape mismatch for key {key}")

            # Formula: (model_1 - base_model) * scaling_factor + model_2
            task_vector = tensor_1 - base_tensor
            scaled_task_vector = task_vector * scaling_factor
            merged_tensor = scaled_task_vector + tensor_2
            merged_state_dict[key] = merged_tensor

        return merged_state_dict