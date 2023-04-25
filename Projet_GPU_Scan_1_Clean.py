import math

import numpy as np
from numba import cuda


@cuda.jit
def scanKernel(array, n):
    m = round(math.log2(n))
    index = cuda.grid(1)


    for d in range(0, m):
        k = index * pow(2, d + 1)
        if k < n - 1:
            array[k + pow(2, d + 1) - 1] += array[k + pow(2, d) - 1]
        cuda.syncthreads()

    array[n - 1] = 0
    cuda.syncthreads()

    for d in range(m - 1, -1, -1):
        k = index * pow(2, d + 1)
        if k < n - 1:
            t = array[k + pow(2, d) - 1]
            array[k + pow(2, d) - 1] = array[k + pow(2, d + 1) - 1]
            array[k + pow(2, d + 1) - 1] += t
        cuda.syncthreads()

def scanGPU(array):
    n = len(array)
    if n > 1024:
        exit(1)

    # Copy array to device
    d_a = cuda.to_device(array)

    # Launch kernel
    threads_per_block = n
    scanKernel[1, threads_per_block](d_a, n)

    # Copy result back to host
    array = d_a.copy_to_host()

    print("Array apres la montée et la descente : ", array)


array = np.array([2, 3, 4, 6], dtype=np.int32)
scanGPU(array)
