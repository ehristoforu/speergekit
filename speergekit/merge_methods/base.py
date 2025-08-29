from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch


class BaseMergeMethod(ABC):
    """
    Abstract base class for merge methods.
    """

    @abstractmethod
    def run(
        self,
        parameters: Dict[str, Any],
        models: List[Dict[str, Any]],
        base_model: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs the merge method.

        Args:
            parameters: A dictionary of parameters for the merge method.
            models: A list of models to merge.
            base_model: The base model, if any.

        Returns:
            A dictionary of merged tensors.
        """
        pass