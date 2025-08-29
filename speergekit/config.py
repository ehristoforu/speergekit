from typing import Dict, List, Optional

from pydantic import BaseModel


class InputModel(BaseModel):
    """
    Represents a model to be merged.
    """
    path: str
    parameters: Optional[Dict] = None


class MergeConfiguration(BaseModel):
    """
    Represents the configuration for a merge operation.
    """
    models: List[InputModel]
    base_model: Optional[str] = None
    merge_method: str
    parameters: Optional[Dict] = None
    dtype: Optional[str] = None