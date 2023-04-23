import numpy as np
from numba import cuda


@cuda.jit
def scanKernel(a, n):
    m = int(np.log2(n))
    for d in range(0, m):
        for k in range(0, n, pow(2, d + 1)):
            idx = k + pow(2, d + 1) - 1
            if idx < n:
                a[idx] += a[k + pow(2, d) - 1]
    cuda.syncthreads()

    a[n - 1] = 0
    cuda.syncthreads()

    for d in range(m - 1, -1, -1):
        for k in range(0, n, pow(2, d + 1)):
            idx1 = k + pow(2, d) - 1
            idx2 = k + pow(2, d + 1) - 1
            if idx2 < n:
                t = a[idx1]
                a[idx1] = a[idx2]
                a[idx2] += t
        cuda.syncthreads()


def scanGPU(a):
    n = len(a)
    if n > 1024:
        raise ValueError("Array size is too large for a single thread block")

    # Copy array to device
    d_a = cuda.to_device(a)

    # Launch kernel
    threads_per_block = n
    scanKernel[1, threads_per_block](d_a, n)

    # Copy result back to host
    a = d_a.copy_to_host()

    print("Array apres la montée et la descente : ", a)


array = np.array([2, 3, 4, 6], dtype=np.int32)
scanGPU(array)
