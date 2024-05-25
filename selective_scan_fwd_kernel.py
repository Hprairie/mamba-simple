import torch
from dataclasses import dataclass
from .kernel_object import threadIdx, blockIdx
from .selective_scan_header import SelectiveScanParams
from .selective_scan_common import load_input, store_input, load_weight
from .selective_scan_common import SSMScanOp, SSMScanPrefixCallbackOp
from .cub import GPUPrefixScan


# ------------------- Selective_scan_fwd_kernel_traits -------------------
# Since python is dynamically typed, there aren't templates like in c++. Thus this is a fake template (think of this like a function)
@dataclass
class Selective_Scan_fwd_kernel_traits:
    kNThreads: int
    kNItems: int
    kIsEvenLen: bool
    isVariableB: bool = True
    isVariableC: bool = True
    kHasZ: bool = True

    input_t: torch.dtype | None = None
    weight_t: torch.dtype | None = None

    # In the original code, we need to determine the amount of shared memory which will be used by each block in the kernel.
    # In order to do a selective scan, each block needs some memory set aside to communicate with other threads within the block.
    # The shared memory is partitioned into several parts, i/o, BlockScan, running_prefix.
    # They calculate the offset of each parition.
    BlockLoadInputT_mem: int = 0
    BlockLoadWeightT_mem: int = 0
    BlockStoreInputT_mem: int = 0
    BlockScanT_mem: int = 0

    # For example given that i/o take up 2 bytes, block scan take up 4 bytes, and running_prefix takes up 4 bytes,
    # then we would have a shared memory size of 10 bytes, and would have an offset of the following
    kSmemIOSize: int = max(
        BlockLoadInputT_mem,
        BlockLoadWeightT_mem * (isVariableB + isVariableC),
        BlockStoreInputT_mem,
    )
    kSmemSize: int = kSmemIOSize + BlockScanT_mem
    # This would capture the offset for each partition

    # In the original code, they look at this size of the pointer, for simplicity we'll just set it to false
    isComplex: bool = False

    BlockScanT = GPUPrefixScan  # The block scan type


#  ------------------- selective_scan_fwd_kernel -------------------


# Since python is dynamically typed, there aren't templates like in c++. Thus this is a fake template (think of this like a function)
@dataclass
class selective_scan_fwd_kernel:
    Ktraits: Selective_Scan_fwd_kernel_traits

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

    def __call__(self, params: SelectiveScanParams):
        kIsComplex = self.Ktraits.isComplex
        kIsVariableB = self.Ktraits.isVariableB
        kIsVariableC = self.Ktraits.isVariableC
        kHasZ = self.Ktraits.kHasZ
        kNThreads = self.Ktraits.kNThreads
        kNItems = self.Ktraits.kNItems

        batch_id = self.blockIdx.x
        dim_id = self.blockIdx.y

        # Even though we don't need shared memory, I will fake create it here to show
        # how this looks like in memory
        smem_load = self.smem_[: self.Ktraits.BlockLoadInputT_mem]
        # If we have both B and C as variable, then we store them in different parts of shared memory
        # so that we don't have to __syncthreads() between loading them
        smem_load_weight = self.smem_[: self.Ktraits.BlockLoadWeightT_mem]
        smem_load_weight1 = self.smem_[
            self.Ktraits.BlockLoadWeightT_mem : 2 * self.Ktraits.BlockLoadWeightT_mem
        ]
        smem_store = self.smem_[: self.Ktraits.BlockStoreInputT_mem]
        smem_scan = self.smem_[self.Ktraits.kSmemIOSize : self.Ktraits.kSmemSize]

        smem_running_prefix = self.smem_[self.Ktraits.kSmemSize :]

        # This isn't real python and would throw an error, but we are mimicking pointers in c++
        u = params.u[batch_id, dim_id]  # This is now a 1D tensor [seqLen]
        delta = params.delta[batch_id, dim_id]  # This is now a 1D tensor [seqLen]
        A = params.A[dim_id]  # This is now a 1D tensor [dstate] (dim of hidden)
        B = params.B[dim_id]  # This is now a 1D tensor [dstate] (dim of hidden)
        Bvar = params.B[batch_id]  # This is now a 2d tensor [dstate, seqLen]
        C = params.C[dim_id]  # This is now a 1D tensor [dstate] (dim of hidden)
        Cvar = params.C[batch_id]  # This is now a 2d tensor [dstate, seqLen]
        x = params.x[batch_id, dim_id]  # This is now a 1D tensor [chunks]

        # The difference between B and Bvar is if we are being selective, for example, if B is static,
        # meaning that it isn't selective, then we get the same value for all sequences in the batch and care about B. However,
        # if it is selective then we get a different value for each sequence in the batch, which would be stored
        # in Bvar. The same goes for C and Cvar.

        # Only one of B or Bvar will be used, depending on the variable kIsVariableB
        # Same goes for C and Cvar, with kIsVariableC

        # We then get D and delta_bias if they exist
        D_val = [0]
        if params.D is not None:
            D_val = params.D[dim_id]

        delta_bias_val = [0]
        if params.delta_bias is not None:
            delta_bias_val = params.delta_bias[dim_id]

        # Selective scan code
        kChunkSize = self.Ktraits.kNThreads * self.Ktraits.kNItems
        for chunk in range(0, params.n_chunks, 1):
            # We create temp variables to store the values of u and delta
            u_vals = torch.Tensor([[0] * kNItems] * kNThreads)
            delta_vals_load = torch.Tensor([[0] * kNItems] * kNThreads)

            # Load the input and delta values into the temp variables
            load_input(u, u_vals, smem_load, params.seqLen - chunk * kChunkSize)
            load_input(
                delta,
                delta_vals_load,
                smem_load_weight,
                params.seqLen - chunk * kChunkSize,
            )

            delta_vals = torch.Tensor([[0] * kNItems] * kNThreads)  # delta
            delta_u_vals = torch.Tensor([[0] * kNItems] * kNThreads)  # delta * u
            out_vals = torch.Tensor([[0] * kNItems] * kNThreads)  # The output values
            # We process delta in this for loop
            # delta_u_vals is used to multiply with B
            # delta_vals will be used to multiply with A
            # out_vals will calculate D * u (the skip connect)
            for i in range(0, kNItems, 1):
                delta_vals[i] = delta_vals_load[i] + delta_bias_val
                if params.softplus:
                    delta_vals[i] = torch.nn.functional.softplus(delta_vals[i])

                delta_u_vals[i] = delta_vals[i] * u_vals[i]
                out_vals[i] = D_val[i] * u_vals[i]

            # We then go through each hidden state and calculate the output
            # Since A is a diagonal matrix we can do this
            for state_idx in range(0, params.dstate, 1):
                A_val = A[state_idx]

                # First we grab B and C
                BC_val = torch.Tensor([0])
                B_vals = torch.Tensor([0] * kNItems)
                C_vals = torch.Tensor([0] * kNItems)

                # When B and C are not variable, then we can fuse them together
                # and reduce computation

                if kIsVariableB:
                    load_weight(
                        Bvar[state_idx],
                        B_vals,
                        smem_load_weight,
                        params.seqLen - chunk * kChunkSize,
                    )  # We use the i/o shared memory
                    if not kIsVariableC:
                        BC_val = C[state_idx]
                if kIsVariableC:
                    smem_load_weight_c = (
                        smem_load_weight1 if kIsVariableB else smem_load_weight
                    )
                    load_weight(
                        Cvar[state_idx],
                        C_vals,
                        smem_load_weight_c,
                        params.seqLen - chunk * kChunkSize,
                    )  # We use a different i/o shared memory for C if B is variable
                    # The reason is so that we don't have to run a __syncthreads() between loading B and C
                    if not kIsVariableB:
                        BC_val = B[state_idx]
                if not kIsVariableB and not kIsVariableC:
                    BC_val = B[state_idx] * C[state_idx]

                # We will now prepare for our prefix scan
                thread_data = torch.Tensor([[0] * 2] * kNItems)
                for i in range(0, kNItems, 1):
                    if not kIsComplex:
                        thread_data[i][0] = delta_vals[i] * A_val
                        thread_data[i][1] = (
                            delta_u_vals[i] * B_vals[i]
                            if kIsVariableB
                            else delta_u_vals[i]
                        )

                        # This essentially zero's out values that are out of bounds
                        if (
                            self.threadIdx.x * kNItems + i
                            >= params.seqLen - chunk * kChunkSize
                        ):
                            thread_data[i][0] = 1
                            thread_data[i][1] = 0
                    else:
                        # Same thing as above but with complex
                        pass

                running_prefix = [None, None]
                if not kIsComplex:
                    # The threadIdx.x % 32 == 0 is an implementation detail
                    # Pretend that we are simple getting the prefix for the entire
                    # chunk that we are about to process
                    running_prefix = (
                        [
                            smem_running_prefix[2 * state_idx],
                            smem_running_prefix[2 * state_idx + 1],
                        ]
                        if chunk > 0 and threadIdx.x % 32 == 0
                        else [1, 0]
                    )
                else:
                    # Same thing as above but with complex
                    pass

                prefix_op = SSMScanPrefixCallbackOp(running_prefix)
                # Read cub Documentation to understand what this is doing
                # Specifically cub::BlockScan
                self.Ktraits.BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp, prefix_op
                )

                # Save the smem_running_prefix for next chunk
                if threadIdx.x == 0:  # So only one thread stores the value
                    smem_running_prefix[2 * state_idx] = prefix_op.running_prefix[0]
                    smem_running_prefix[2 * state_idx + 1] = prefix_op.running_prefix[1]

                    # Store the aggregate value for the entire chunk
                    # This is used in the backward pass to optimize the computation
                    # for calculating the gradients of delta
                    x[chunk, state_idx][0] = prefix_op.running_prefix[0]
                    x[chunk, state_idx][1] = prefix_op.running_prefix[1]

                # multiply by C and write to output
                for i in range(0, kNItems, 1):
                    if kIsVariableC:
                        # if B is not variable then we fuse it in here
                        C_val = C_vals[i] if kIsVariableB else C_vals[i] * BC_val
                    else:
                        C_val = BC_val
                    if not kIsComplex:
                        out_vals[i] += thread_data[i][0] * C_val
                    else:
                        # Same thing as above but with complex
                        pass

            # After doing all state_idx
            # This represents a pointer to the output
            out = params.out[batch_id, dim_id]
            # Save out_vals at the pointer of out
            store_input(out, out_vals, smem_store, params.seqLen - chunk * kChunkSize)

            # If we have a Z, we go ahead and calculate it and store it
            if kHasZ:
                # These represent pointers to z and z_out
                z = params.z[batch_id, dim_id]
                z_out = params.z_out[batch_id, dim_id]

                # Temp storage for our specific thread
                z_vals = torch.Tensor([0] * kNItems)
                load_input(z, z_vals, smem_load, params.seqLen - chunk * kChunkSize)
                # Apply non linearity for each item our thread gets
                for i in range(0, kNItems, 1):
                    # Sigmoid non linearity
                    out_vals[i] = z_vals[i] / (1 + torch.exp(-out_vals[i]))
                # Store the output item from our thread
                store_input(
                    z_out, out_vals, smem_store, params.seqLen - chunk * kChunkSize
                )


#  ------------------- selective_scan_fwd_launch -------------------


# Since python is dynamically typed, there aren't templates like in c++. Thus this is a fake template (think of this like a function)
@dataclass
class selective_scan_fwd_launch:
    kNThreads: int  # The number of threads each kernel gets
    kNItems: int  # The number of items that each thread will process

    # For those unfaimilair with c++, since it's a static language, we need to pass the types of input and weight's
    input_t: torch.dtype
    weight_t: torch.dtype

    def __call__(self, params: SelectiveScanParams):
        kIsEvenLen = params.seqLen % (self.kNThreads * self.kNItems) == 0
        isVariableB = params.isVariableB
        isVariableC = params.isVariableC
        kHasZ = params.z is not None

        # Create the kernel traits
        Ktraits = Selective_Scan_fwd_kernel_traits(
            self.kNThreads,
            self.kNItems,
            kIsEvenLen,
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
        selective_scan_fwd_kernel(Ktraits)(params)

        # launch the kernel would be like this:
        # selective_scan_fwd_kernel<Ktraits><<<grid, kNThreads, mem_size, stream>>>(params)


#  ------------------- selective_scan_fwd_cuda -------------------
# Since python is dynamically typed, there aren't templates like in c++. Thus this is a fake template (think of this like a function)
@dataclass
class selective_scan_fwd_cuda:
    # For those unfaimilair with c++, since it's a static language, we need to pass the types of input and weight's
    input_t: torch.dtype
    weight_t: torch.dtype

    def __call__(self, params: SelectiveScanParams):
        """
        There is no need to launch a kernel with a ton of threads if the sequence length is small.

        Thus we have a few different kernels for different sequence lengths.
        """
        if params.seqLen <= 128:
            selective_scan_fwd_launch(32, 4, self.input_t, self.weight_t)(params)
        elif params.seqLen <= 256:
            selective_scan_fwd_launch(32, 8, self.input_t, self.weight_t)(params)
        elif params.seqLen <= 512:
            selective_scan_fwd_launch(32, 16, self.input_t, self.weight_t)(params)
        elif params.seqLen <= 1024:
            selective_scan_fwd_launch(64, 16, self.input_t, self.weight_t)(params)
        else:
            selective_scan_fwd_launch(128, 16, self.input_t, self.weight_t)(params)
