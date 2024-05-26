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

        # ----- Shared memory for constants -----

        # Go down to where we are calculate delta * A to see how and why we are using this
        smem_delta_a = self.smem_[
            self.Ktraits.kSmemSize : self.Ktraits.kSmemSize
            + kNThreads
            + 2 * params.dstate
        ]

        # postfix memory
        smem_running_postfix = self.smem_  # I got lazy here but you get the point
        # This will store params.dstate amount of values, which are the postfix values for the next block

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

            # Load values into the temporary variables (These are thread specific)
            load_input(u, u_vals, smem_load, params.seqLen - chunk * kNItems)
            load_input(
                delta, delta_vals_load, smem_load, params.seqLen - chunk * kNItems
            )
            load_input(dout, dout_vals_load, smem_load, params.seqLen - chunk * kNItems)

            # Create more temporary variables to store delta and dout
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

                # Load values into the temporary variables
                load_input(z, z_vals, smem_load, params.seqLen - chunk * kNItems)
                # __syncthreads() # We don't want to overwrite smem_load memory
                load_input(out, out_vals, smem_load, params.seqLen - chunk * kNItems)

                # Temporary variables to store the values of dz and z_silu on the thread
                dz_vals = torch.Tensor([0] * kNItems)
                # We will reuse z_silu_vals that's why we store them here
                z_silu_vals = torch.Tensor([0] * kNItems)
                for i in range(kNItems):
                    # Get the z val for each item in the thread and calculate it's sigmoid
                    z_val = z_vals[i]
                    z_sigmoid_val = 1 / (1 + torch.exp(-z_val))

                    # Calculate the dz value
                    # Remember that the derivative of the SiLU function is: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                    # Thus dout/dz = out * sigmoid(z) * (1 + z * (1 - sigmoid(z)))
                    z_silu_vals[i] = z_val * z_sigmoid_val
                    dz_vals[i] = (
                        dout_vals[i]
                        * out_vals[i]
                        * z_sigmoid_val
                        * (1 + z_val * (1 - z_sigmoid_val))
                    )

                    # Also note that since silu is out * SiLU(z), then for gradients to flow
                    # though dout, we need to multiply by SiLU(z) as well. (Think gating function)
                    dout_vals[i] *= z_silu_vals[i]

                # Now that we calculated dz vals, we can store them
                store_input(dz, dz_vals, smem_store, params.seqLen - chunk * kNItems)

                # Why do they recalculate out_z_vals here?

            # Before we get started on calculating the gradients throught the backward pass,
            # if we have a D value, we need to calculate the gradients for it, and then also
            # pass gradients thought to u

            # This will be used to store the gradients for u
            du_vals = torch.Tensor([0] * kNItems)

            for i in range(kNItems):
                du_vals[i] = dout_vals[i] * D_val
                dD_val += dout_vals[i] * u_vals[i]
                # Note as D is not selective, we accumulate the gradients for it

            # Create a temporary store for the gradients of delta
            # We use this at the end of the loop
            ddelta_vals = torch.Tensor([0] * kNItems)

            # Like the forward process of selective scan, since A is diagonal, then
            # we will calculate the gradients for each rank independently
            for stateIdx in range(params.dstate):
                A_val = A[stateIdx]  # Grab the A value for this rank

                # Create temporary local storage B_vals and C_vals for this rank
                B_val, C_val = 0, 0
                B_vals = torch.Tensor([0] * kNItems)
                C_vals = torch.Tensor([0] * kNItems)

                # Load the B and C values for this rank
                if kIsVariableB:
                    load_weight(Bvar[stateIdx], B_vals, smem_load_weight, 0)
                else:
                    B_val = B[stateIdx]

                if kIsVariableC:
                    load_weight(Cvar[stateIdx], C_vals, smem_load_weight1, 0)
                else:
                    C_val = C[stateIdx]

                # These will be used to store values which we are going to run a
                # selective scan on
                # We will preprocess gradient streams and then load the values to be
                # scanned into these parameters. Then just run selective scan
                thread_data = torch.Tensor([0] * kNItems)
                thread_reverse_data = torch.Tensor([0] * kNItems)
                if kIsComplex:
                    # In order to calculate the gradients for delta, we need the hidden state
                    # This is why we are doing two selective scans. The first one is a prefix
                    # scan which will be used to recalculate the hidden states, and the second
                    # one is a postfix scan which will be used to calculate the gradients for everything else

                    # Let's first load the correct values into thread_data
                    # and thread_reverse_data
                    for i in range(kNItems):
                        # We need to recalculate exp(delta * A) for each item
                        delta_a_exp = torch.exp(delta_vals[i] * A_val)

                        # This is the same as the forward pass
                        thread_data[i][0] = delta_vals[i] * A_val
                        thread_data[i][1] = (
                            delta_vals[i] * B_vals[i] * u_vals[i]
                            if kIsVariableB
                            else delta_vals[i] * u_vals[i]
                        )

                        # We have the data loaded into thread_data, now we need to load data
                        # into reverse thread data.

                        # First recognize that the backward pass will look something like this for a sequence of length 3
                        # dh_0 = dout_0 + dout_1 * A_1 + dout_2 * A_2 * A_1
                        # dh_1 = dout_1 + dout_2 * A_2
                        # dh_2 = dout_2

                        # We can see that this is just the same selective scan as the forward pass, but in the reverse order
                        # of the sequence. One maybe not so obvious difference is that we need to shift the values of A to
                        # the left by one for the reverse scan in comparison to the forward scan.

                        # I would take a close look at what is in the values of both the forward and reverse scan's if you are confused

                        if i == 0:
                            smem_delta_a[
                                (
                                    stateIdx + (chunk % 2) * params.dstate
                                    if threadIdx.x == 0
                                    else threadIdx.x + 2 * params.dstate
                                )
                            ] = delta_a_exp
                        else:
                            thread_reverse_data[i - 1][0] = delta_a_exp

                        # Alright I'll try to explain what is happening in the above snippet. Now that we know that we need to shift
                        # the values of A to the left by one for the reverse scan, it would be much better to use shared memory to
                        # shift the values of A to the left, rather then going to memory.

                        # The memory layout of smem_delta_a is as follows:
                        # Section 1: size is (2 * params.dstate)
                        #    - Here we store the values of exp(delta * A) which are stored in threadIdx.x == 0
                        #    - This should be the A values that are needed in the next chunk
                        #    - So rather than recompute them, we store them in shared memory
                        #    - The modulo operation is used to switch between two sections of memory
                        #      so that we do not overite our current values as they will be needed in our current chunk
                        #    - Thus we have an active section which our current chunk is using and a future chunk,
                        #      which the next chunk will use
                        # Section 2: size is the number of threads in our block
                        #    - Here we store the values of exp(delta * A) which we will use to shift for our current block

                        # We then calculate y for the selective scan (pretty standard)
                        if kIsVariableC:
                            thread_reverse_data[i][1] = dout_vals[i] * (
                                C_vals[i] if kIsVariableB else C_vals[i] * B_val
                            )
                        else:
                            thread_reverse_data[i][1] = dout_vals[i] * (
                                C_val if kIsVariableB else C_val * B_val
                            )
                    # __syncthreads();
                    # ^^^^ This is needed here to that we have fully written to shared memory before we start reading from it

                    # We then preform the shuffle to the left
                    if threadIdx.x == kNThreads - 1:
                        if chunk == params.n_chunks - 1:
                            thread_reverse_data[kNItems - 1][0] = 1.0
                        else:
                            thread_reverse_data[kNItems - 1][0] = smem_delta_a[
                                stateIdx + ((chunk + 1) % 2) * params.dstate
                            ]
                    else:
                        thread_reverse_data[kNItems - 1][0] = smem_delta_a[
                            threadIdx.x + 1 + 2 * params.dstate
                        ]

                    # Now we preform our selective scans
                    if chunk > 0 and threadIdx.x % 32 == 0:
                        # We had saved x from the forward pass (very cheap optimization with huge benefits)
                        running_prefix = x[(chunk - 1) * params.dstate + stateIdx]
                    else:
                        running_prefix = [1, 0]

                    # Forward scan for hidden states
                    prefix_op = SSMScanPrefixCallbackOp(running_prefix)
                    self.Ktraits.BlockScanT(smem_scan).InclusiveScan(
                        thread_data, thread_data, SSMScanOp, prefix_op
                    )

                    # Reverse scan for gradient propagation
                    if chunk < params.n_chunks - 1 and threadIdx.x == 0:
                        running_postfix = smem_running_postfix[stateIdx]
                    else:
                        running_postfix = [1, 0]
                    postfix_op = SSMScanPrefixCallbackOp(running_postfix)
                    self.Ktraits.BlockScanReverseT(smem_reverse_scan).InclusiveScan(
                        thread_reverse_data, thread_reverse_data, SSMScanOp, postfix_op
                    )

                    # Save the running postfix for the next block
                    if threadIdx.x == 0:
                        smem_running_postfix[stateIdx] = postfix_op.running_prefix

                    # Alright now its time to calculate the gradient flow to the rest of our parameters
                    # and our output values

                    # We will start by creating temporary variables which our specific to this thread
                    dA_val = 0
                    dBC_val = 0  # This will not be used if B and C are both selective
                    dB_vals = torch.Tensor([0] * kNItems)
                    dC_vals = torch.Tensor([0] * kNItems)
                    for i in range(kNItems):
                        dx = thread_reverse_data[i][1]
                        # If B isn't variable then it was already added in with BC_var
                        # This is the derivative of the hidden state with respect to delta * x
                        ddelta_u = dx if not kIsVariableB else dx * B_vals[i]

                        # We can then get the gradients for u from d (delta * u)
                        du_vals[i] += ddelta_u * delta_vals[i]

                        # Now we want to calculate the gradients for delta
                        # They come from two sources, the expression (delta * u) and also delta * A
                        # For the second note that exp(delta * A) * x + B (zeta), where zeta = delta * u
                        # Thus when taking the expression with respect to delta, we get A * exp(delta * A) * x

                        # Here a is just our previous hidden state
                        if kIsVariableB:
                            a = thread_data[i][1] - (
                                B_vals[i] * u_vals[i] * delta_vals[i]
                            )
                        else:
                            a = thread_data[i][1] - (u_vals[i] * delta_vals[i])

                        # You should recognize that this is what I explained above
                        ddelta_vals[i] = ddelta_u * u_vals[i] + dx * A_val * a

                        # The same goes for A as well
                        # This is just like the second part of delta
                        dA_val += dx * delta_vals[i] * a

                        # Now we calculate the gradients for B and C
                        if not kIsVariableB or not kIsVariableC:
                            pass

                else:
                    # The same thing as the real case but with complex numbers
                    pass


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
