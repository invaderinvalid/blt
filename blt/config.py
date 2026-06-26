from dataclasses import dataclass


@dataclass
class GlobalConfig:

    # number of transformer blocks
    n_layer: int = 12

    # attention heads
    n_head: int = 12

    # embedding dimension
    # MUST match patch dimension
    n_embd: int = 768

    # maximum number of patches
    block_size: int = 512