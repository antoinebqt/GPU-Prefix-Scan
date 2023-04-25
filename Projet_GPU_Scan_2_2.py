import math

import numpy as np
from numba import cuda


@cuda.jit
def scanKernel(array, n):
    m = round(math.log2(n))
    k = cuda.grid(1)
    if k == 0:
        print("n :", n)
        print("m :", m)

    shared_array = cuda.shared.array(shape=1024, dtype=np.int32)

    shared_array[k] = array[k]
    cuda.syncthreads()

    for d in range(0, m):
        cuda.syncthreads()
        step = 2 ** (d + 1)
        if k >= n or k % step != 0 and k != 0:
            continue
        shared_array[k + pow(2, d + 1) - 1] += shared_array[k + pow(2, d) - 1]

    cuda.syncthreads()
    shared_array[n - 1] = 0
    cuda.syncthreads()

    for d in range(m - 1, -1, -1):
        cuda.syncthreads()
        step = 2 ** (d + 1)
        if k >= n or k % step != 0 and k != 0:
            continue
        t = shared_array[k + pow(2, d) - 1]
        shared_array[k + pow(2, d) - 1] = shared_array[k + pow(2, d + 1) - 1]
        shared_array[k + pow(2, d + 1) - 1] += t

    cuda.syncthreads()

    array[k] = shared_array[k]


def scanGPU(array):
    n = len(array)
    if n > 1024:
        exit(1)

    m = int(math.pow(2, math.ceil(math.log2(n))))  # Puissance de 2 supérieure

    # Create a new array of size m with the original array followed by zeros
    padded_array = np.zeros(m, dtype=np.int32)
    padded_array[:n] = array

    # Copy array to device
    d_a = cuda.to_device(padded_array)

    # Launch kernel
    threads_per_block = m
    print("Threads par blocks :", threads_per_block)
    scanKernel[1, threads_per_block](d_a, m)

    # Copy result back to host
    res = d_a.copy_to_host()[:n]

    print("Array apres la montée et la descente : ", res)


array = np.array([2, 3, 4, 6, 1, 2, 3], dtype=np.int32)
scanGPU(array)
