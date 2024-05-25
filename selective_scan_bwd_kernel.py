import torch
from dataclasses import dataclass
from .kernel_object import threadIdx, blockIdx
from .selective_scan_header import SSMParamsBwd
from .selective_scan_common import load_input, store_input, load_weight
from .selective_scan_common import SSMScanOp, SSMScanPrefixCallbackOp
from .cub import GPUPrefixScan
from .reverse_scan import GPUPostfixScan


# ------------------- Selective_scan_bwd_kernel_traits -------------------
# Since python is dynamically typed, there aren't templates like in c++. Thus this is a fake template (think of this like a function)
@dataclass
class Selective_Scan_bwd_kernel_traits:
    kNThreads: int
    kNItems: int
    kIsEvenLen: bool
    KDeltaSoftplus: bool
    isVariableB: bool = True
    isVariableC: bool = True
    kHasZ: bool = True

    input_t: torch.dtype | None = None
    weight_t: torch.dtype | None = None

    # In the original code, we need to determine the amount of shared memory which will be used by each block in the kernel.
    # In order to do a selective scan, each block needs some memory set aside to communicate with other threads within the block.
    # The shared memory is partitioned into several parts, i/o, BlockScan, BlockExchange, BlockReduce, and running_prefix.
    # They calculate the offset of each parition.
    BlockLoadInputT_mem: int = 0
    BlockLoadWeightT_mem: int = 0
    BlockStoreInputT_mem: int = 0
    BlockScanT_mem: int = 0
    BlockScanReverseT_mem: int = 0
    BlockExchangeT_mem: int = 0
    BlockReduceT_mem: int = 0

    # For example given that i/o take up 2 bytes, block scan take up 6 bytes, reverse block scan take up to 7 bytes,
    # block reduce take 3 bytes, and block exchange takes 4 bytes, and running_prefix takes 12 bytes
    # then we would have a shared memory size of 34 bytes, and would have an offset of the following
    kSmemIOSize: int = max(
        BlockLoadInputT_mem,
        BlockLoadWeightT_mem * (isVariableB + isVariableC),
        BlockStoreInputT_mem,
    )
    kSmemExchangeSize: int = (isVariableB + isVariableC) * BlockExchangeT_mem
    kSemReduceSize: int = BlockReduceT_mem
    kSmemSize: int = (
        kSmemIOSize
        + kSmemExchangeSize
        + kSemReduceSize
        + BlockScanT_mem
        + BlockScanReverseT_mem
    )
    # This would capture the offset for each partition
    # If we allocated 34 bytes of shared memory, then the running prefix would be [kSmemSize:] of the shared memory

    # In the original code, they look at this size of the pointer, for simplicity we'll just set it to false
    isComplex: bool = False

    BlockScanT = GPUPrefixScan  # The block scan type
    BlockScanReverseT = GPUPostfixScan  # The reverse block scan reverse type


#  ------------------- selective_scan_bwd_kernel -------------------


# Since python is dynamically typed, there aren't templates like in c++. Thus this is a fake template (think of this like a function)
@dataclass
class selective_scan_bwd_kernel:
    Ktraits: Selective_Scan_bwd_kernel_traits

    threadIdx = threadIdx(0, 0, 0)

    # The blockIdx will tell use which Batch and Dim we are working on in this kernel
    # i.e. for (5, 2, 1) we will be working on the 5th sequence in the batch and specifically the 2nd dim of that sequence
    blockIdx = blockIdx(0, 0, 0)

    # CUDA stream
    stream = None

    # Shared memory allocation
    mem_size: int = (
        0  # This would not really be zero, and would be set by the kernel launch traits
    )

    def __post__init__(self):
        # Allocate the shared memory for the block
        self.smem_ = torch.zeros(self.mem_size)
        # While this class is a kernel, we need to remember that shared memory is shared across all threads in a block

    def __call__(self, params: SSMParamsBwd):
        kIsComplex = self.Ktraits.isComplex
        kIsVariableB = self.Ktraits.isVariableB
        kIsVariableC = self.Ktraits.isVariableC
        kHasZ = self.Ktraits.kHasZ
        kNThreads = self.Ktraits.kNThreads
        kNItems = self.Ktraits.kNItems

        batch_id = self.blockIdx.x
        dim_id = self.blockIdx.y

        # Create temp store for weights
        # In CUDA, we need some shared memory across all threads in a block to either preform
        # operations or store values that are shared across all threads
        smem_load = self.smem_[: self.Ktraits.BlockLoadInputT_mem]
        smem_load_weight = self.smem_[: self.Ktraits.BlockLoadWeightT_mem]
        smem_load_weight1 = self.smem_[
            self.Ktraits.BlockLoadWeightT_mem : 2 * self.Ktraits.BlockLoadWeightT_mem
        ]
        smem_store = self.smem_[: self.Ktraits.BlockStoreInputT_mem]
        smem_exchange = self.smem_[
            self.Ktraits.kSmemIOSize : self.Ktraits.kSmemExchangeSize
        ]
        smem_exchange1 = self.smem_[
            self.Ktraits.kSmemIOSize
            + self.Ktraits.BlockExchangeT_mem : self.Ktraits.kSmemIOSize
            + 2 * self.Ktraits.BlockExchangeT_mem
        ]
        smem_reduce = self.smem_[
            self.Ktraits.kSmemIOSize
            + self.Ktraits.kSmemExchangeSize : self.Ktraits.kSmemIOSize
            + self.Ktraits.kSmemExchangeSize
            + self.Ktraits.BlockReduceT_mem
        ]

        # I gave up here, but I think you get the point about shared memory
        smem_scan = self.smem_
        smem_reverse_scan = self.smem_

        # Shared memory for constants

        # This isn't real python and would throw an error, but we are mimicking pointers in c++
        u = params.u[batch_id, dim_id]  # This is now a 1D tensor [seqLen]
        delta = params.delta[batch_id, dim_id]  # This is now a 1D tensor [seqLen]
        A = params.A[dim_id]  # This is now a 1D tensor [dstate] (dim of hidden)
        B = params.B[dim_id]  # This is now a 1D tensor [dstate] (dim of hidden)
        Bvar = params.B[batch_id]  # This is now a 2d tensor [dstate, seqLen]
        C = params.C[dim_id]  # This is now a 1D tensor [dstate] (dim of hidden)
        Cvar = params.C[batch_id]  # This is now a 2d tensor [dstate, seqLen]
        x = params.x[batch_id, dim_id]  # This is now a 1D tensor [chunks]

        # Backward pointers
        dout = params.dout[batch_id, dim_id]  # This is now a 1D tensor [seqLen]
        dA = params.dA[dim_id]  # This is now a 1D tensor [dstate] (dim of hidden)
        # dB is either a 1d tensor [dstate] or a 2d tensor [dstate, seqLen]
        # This is determined by if B is selective
        dB = params.dB[dim_id] if not kIsVariableB else params.dB[batch_id]
        # The same for dC, it is either a 1d tensor [dstate] or a 2d tensor [dstate, seqLen]
        dC = params.dC[dim_id] if not kIsVariableC else params.dC[batch_id]
        dD = params.dD[dim_id]  # This is a scalar value if it exists

        dD_val = 0
        dDelta_bias = 0

        # We then get D and delta_bias if they exist
        D_val = [0]
        if params.D is not None:
            D_val = params.D[dim_id]

        delta_bias_val = [0]
        if params.delta_bias is not None:
            delta_bias_val = params.delta_bias[dim_id]

        # Iterate through the sequence a chunk at a time
        for chunk in range(params.n_chunks - 1, 0, -1):
            # Create temporary variables to store the values of u, delta, a dout,
            # We need these to stream infomation to our current kernel
            u_vals = torch.Tensor([0] * kNItems)
            delta_vals_load = torch.Tensor([0] * kNItems)
            dout_vals_load = torch.Tensor([0] * kNItems)

            load_input(u, u_vals, smem_load, params.seqLen - chunk * kNItems)
            load_input(
                delta, delta_vals_load, smem_load, params.seqLen - chunk * kNItems
            )
            load_input(dout, dout_vals_load, smem_load, params.seqLen - chunk * kNItems)

            dout_vals = torch.Tensor([0] * kNItems)
            delta_vals = torch.Tensor([0] * kNItems)
            for i in range(kNItems):
                # Here we are up casting to fp32 so that we have enough precision
                dout_vals[i] = float(dout_vals_load[i])
                # We also recalculate delta values since we didn't save them in the forward pass
                delta_vals[i] = delta_vals_load[i] + delta_bias_val
                if params.softplus:
                    delta_vals[i] = torch.nn.functional.softplus(delta_vals[i])

            # We calculate the Z values if they exist before we do anything else
            # This is because the Z values are often fused with the kernel
            # Thus gradients need to pass through them first
            if kHasZ:
                z = params.z[batch_id, dim_id]
                out = params.out[batch_id, dim_id]
                dz = params.dz[batch_id, dim_id]

                # Temporary variables to store the values of z and out on the thread
                z_vals = torch.Tensor([0] * kNItems)
                out_vals = torch.Tensor([0] * kNItems)


#  ------------------- selective_scan_bwd_launch -------------------


# Since python is dynamically typed, there aren't templates like in c++. Thus this is a fake template (think of this like a function)
@dataclass
class selective_scan_bwd_launch:
    kNThreads: int  # The number of threads each kernel gets
    kNItems: int  # The number of items that each thread will process

    # For those unfaimilair with c++, since it's a static language, we need to pass the types of input and weight's
    input_t: torch.dtype
    weight_t: torch.dtype

    def __call__(self, params: SSMParamsBwd):
        kIsEvenLen = params.seqLen % (self.kNThreads * self.kNItems) == 0
        isVariableB = params.isVariableB
        isVariableC = params.isVariableC
        kHasZ = params.z is not None
        kDeltaSoftplus = params.softplus

        # Create the kernel traits
        Ktraits = Selective_Scan_bwd_kernel_traits(
            self.kNThreads,
            self.kNItems,
            kIsEvenLen,
            kDeltaSoftplus,
            isVariableB,
            isVariableC,
            kHasZ,
            self.input_t,
            self.weight_t,
        )

        # Calculate the amount of shared memory that will be used by each block
        mem_size = Ktraits.kSmemSize + Ktraits.kNItems
        # ^^^^ This is essentially what is happening without some pointer math

        grid = (params.batch, params.dim, 1)  # The Launch grid for cuda
        selective_scan_bwd_kernel(Ktraits)(params)

        # launch the kernel would be like this:
        # selective_scan_fwd_kernel<Ktraits><<<grid, kNThreads, mem_size, stream>>>(params)


#  ------------------- selective_scan_bwd_cuda -------------------
# Since python is dynamically typed, there aren't templates like in c++. Thus this is a fake template (think of this like a function)
@dataclass
class selective_scan_bwd_cuda:
    # For those unfaimilair with c++, since it's a static language, we need to pass the types of input and weight's
    input_t: torch.dtype
    weight_t: torch.dtype

    # This is the same as selective_scan_fwd_cuda
    def __call__(self, params: SSMParamsBwd):
        """
        There is no need to launch a kernel with a ton of threads if the sequence length is small.

        Thus we have a few different kernels for different sequence lengths.
        """
        if params.seqLen <= 128:
            selective_scan_bwd_launch(32, 4, self.input_t, self.weight_t)(params)
        elif params.seqLen <= 256:
            selective_scan_bwd_launch(32, 8, self.input_t, self.weight_t)(params)
        elif params.seqLen <= 512:
            selective_scan_bwd_launch(32, 16, self.input_t, self.weight_t)(params)
        elif params.seqLen <= 1024:
            selective_scan_bwd_launch(64, 16, self.input_t, self.weight_t)(params)
        else:
            selective_scan_bwd_launch(128, 16, self.input_t, self.weight_t)(params)
