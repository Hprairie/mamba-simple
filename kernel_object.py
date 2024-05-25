from dataclasses import dataclass


@dataclass
class threadIdx:
    """
    Used to represent the thread index in CUDA.
    """

    x: int
    y: int
    z: int


@dataclass
class blockIdx:
    """
    Used to represent the block index in CUDA.
    """

    x: int
    y: int
    z: int
