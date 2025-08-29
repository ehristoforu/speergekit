import logging
from typing import Any, Dict

import torch
import yaml

from speergekit.config import MergeConfiguration
from speergekit.io.lazy_tensor_loader import LazyTensorLoader
from speergekit.io.tensor_writer import TensorWriter
from speergekit.merge_methods.linear import LinearMerge
from speergekit.merge_methods.task_arithmetic import TaskArithmeticMerge

MERGE_METHODS = {
    "linear": LinearMerge,
    "task_arithmetic": TaskArithmeticMerge,
}

def run_merge(
    merge_config: MergeConfiguration,
    out_path: str,
    options: Dict[str, Any],
):
    """
    Performs a merge of models based on the provided configuration.

    Args:
        merge_config: The configuration for the merge.
        out_path: The path to save the merged model.
        options: A dictionary of additional options.
    """
    logging.info(f"Loading {len(merge_config.models)} models")
    models = [
        LazyTensorLoader.from_disk(model.path) for model in merge_config.models
    ]
    base_model = (
        LazyTensorLoader.from_disk(merge_config.base_model)
        if merge_config.base_model
        else None
    )

    merge_method_name = merge_config.merge_method
    if merge_method_name not in MERGE_METHODS:
        raise ValueError(f"Unknown merge method: {merge_method_name}")

    logging.info(f"Merging with method {merge_method_name}")
    merge_method = MERGE_METHODS[merge_method_name]()
    merged_state_dict = merge_method.run(
        parameters=merge_config.parameters,
        models=models,
        base_model=base_model,
    )

    logging.info(f"Saving merged model to {out_path}")
    writer = TensorWriter(out_path)
    for key, value in merged_state_dict.items():
        writer.save_tensor(key, value)
    writer.finalize()

    logging.info("Merge complete")