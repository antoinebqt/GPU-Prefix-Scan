import numpy as np
from numba import cuda, int32


@cuda.jit
def scan_kernel(a):
    # Allocate shared memory for each thread block
    sm = cuda.shared.array(shape=1024, dtype=int32)

    # Load data from global memory into shared memory
    tid = cuda.threadIdx.x
    idx = cuda.blockIdx.x * cuda.blockDim.x + tid
    if idx < a.size:
        sm[tid] = a[idx]
    else:
        sm[tid] = 0

    # Up-sweep phase
    for d in range(0, cuda.shared.array.shape[0].bit_length() - 1):
        stride = 1 << (d + 1)
        if tid % stride == 0:
            sm[tid + stride - 1] += sm[tid + stride // 2 - 1]
        cuda.syncthreads()

    # Clear last element and down-sweep phase
    if tid == 0:
        sm[cuda.shared.array.shape[0] - 1] = 0
    cuda.syncthreads()

    for d in range(cuda.shared.array.shape[0].bit_length() - 2, -1, -1):
        stride = 1 << (d + 1)
        if tid % stride == 0:
            t = sm[tid + stride // 2 - 1]
            sm[tid + stride // 2 - 1] = sm[tid + stride - 1]
            sm[tid + stride - 1] += t
        cuda.syncthreads()

    # Write results back to global memory
    if idx < a.size:
        a[idx] = sm[tid]


def scanGPU(a):
    # Copy data to device
    d_a = cuda.to_device(a)

    # Determine block and grid sizes
    block_size = 1024
    grid_size = (a.size + block_size - 1) // block_size

    # Launch kernel
    scan_kernel[grid_size, block_size](d_a)

    # Copy results back to host
    d_a.copy_to_host(a)


array = np.array([2, 3, 4, 6], dtype=np.int32)
scanGPU(array)
