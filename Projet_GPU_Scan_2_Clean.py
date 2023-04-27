import math

import numpy as np
from numba import cuda


@cuda.jit
def scanKernel(array, n):
    m = round(math.log2(n))
    idx = cuda.grid(1)
    
    if idx == 0:
        print("n :", n)
        print("m :", m)

    shared_array = cuda.shared.array(shape=1024, dtype=np.int32)

    shared_array[idx] = array[idx]
    cuda.syncthreads()

    for d in range(0, m):
        k = idx * pow(2, d + 1)
        if k < n - 1:
            shared_array[k + pow(2, d + 1) - 1] += shared_array[k + pow(2, d) - 1]
        cuda.syncthreads()

    if idx == 0:
        shared_array[n - 1] = 0
    cuda.syncthreads()

    for d in range(m - 1, -1, -1):
        k = idx * pow(2, d + 1)
        if k < n - 1:
            t = shared_array[k + pow(2, d) - 1]
            shared_array[k + pow(2, d) - 1] = shared_array[k + pow(2, d + 1) - 1]
            shared_array[k + pow(2, d + 1) - 1] += t
        cuda.syncthreads()

    array[idx] = shared_array[idx]


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
    scanKernel[1, threads_per_block](d_a, m)

    # Copy result back to host
    array = d_a.copy_to_host()[:n]

    print("Array apres la montée et la descente : ", array)


array = np.array([2, 9, 15, 13, 10, 20, 2, 3, 1, 0], dtype=np.int32)
scanGPU(array)
