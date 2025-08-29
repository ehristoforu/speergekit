from dataclasses import dataclass

@dataclass
class MergeOptions:
    """
    Configuration for merging models.
    """
    cuda: bool = False
    lazy_unpickle: bool = False
    low_cpu_memory: bool = False