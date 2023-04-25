import math

import numpy as np
from numba import cuda, int32


@cuda.jit
def scanKernel(array, n):
    m = round(math.log2(n))
    index = cuda.grid(1)

    shared_array = cuda.shared.array(shape=1024, dtype=int32)

    if index < n:
        shared_array[index] = array[index]
    cuda.syncthreads()

    for d in range(0, m):
        k = index * pow(2, d + 1)
        if k < n - 1:
            shared_array[k + pow(2, d + 1) - 1] += shared_array[k + pow(2, d) - 1]
        cuda.syncthreads()

    if index == 0:
        shared_array[n - 1] = 0
    cuda.syncthreads()

    for d in range(m - 1, -1, -1):
        k = index * pow(2, d + 1)
        if k < n - 1:
            t = shared_array[k + pow(2, d) - 1]
            shared_array[k + pow(2, d) - 1] = shared_array[k + pow(2, d + 1) - 1]
            shared_array[k + pow(2, d + 1) - 1] += t
        cuda.syncthreads()

    array[index] = shared_array[index]
    cuda.syncthreads()


def scanGPU(array):
    n = len(array)
    if n > 1024:
        exit(1)

    # Copy array to device
    d_a = cuda.to_device(array)

    # Launch kernel
    threads_per_block = int(math.pow(2, math.ceil(math.log2(n))))  # Puissance de 2 supérieure
    print("Threads par blocks :", threads_per_block)
    scanKernel[1, threads_per_block](d_a, n)

    # Copy result back to host
    array = d_a.copy_to_host()

    print("Array apres la montée et la descente : ", array)


array = np.array([2, 3, 4, 6, 1], dtype=np.int32)
scanGPU(array)
