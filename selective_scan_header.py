import torch
from dataclasses import dataclass


@dataclass
class SelectiveScanParams:
    batch: int
    dim: int
    seqLen: int
    dstate: int
    n_groups: int
    n_chunks: int

    isVariableB: bool  # Is it selective or not
    isVariableC: bool  # Is it selective or not

    softplus: bool  # Use softplus or not

    # Seleective Scan weights
    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor

    # Input Singal
    u: torch.Tensor

    # Discretization Delta
    delta: torch.Tensor
    delta_bias: torch.Tensor

    # Output Signal to store selective scan output
    out: torch.Tensor

    # Saved hidden states for prefix sum
    # Note: we don't save every hidden state, just the last one of each block
    # We will need this to save time for the backward pass
    x: torch.Tensor

    # Non-linear signal for selective scan, used to fuse with our kernel as we already
    # will have our signal in memory, might as well use it
    z: torch.Tensor
    z_out: torch.Tensor


@dataclass
class SSMParamsBwd(SelectiveScanParams):
    # Incoming gradients
    dout: torch.Tensor

    # Outgoing Gradients
    dA: torch.Tensor
    dB: torch.Tensor
    dC: torch.Tensor
    dD: torch.Tensor

    # Outgoing Signal Gradients
    du: torch.Tensor
    dz: torch.Tensor

    # Outgoing Delta Gradients
    ddelta: torch.Tensor
    ddelta_bias: torch.Tensor
