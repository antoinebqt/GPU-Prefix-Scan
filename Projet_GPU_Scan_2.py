import math

import numpy as np
from numba import cuda


@cuda.jit
def scanKernel(array, n):
    m = round(math.log2(n))
    x = cuda.grid(1)
    if x == 0:
        print("n :", n)
        print("m :", m)

    shared_array = cuda.shared.array(shape=1024, dtype=np.int32)

    if x < n:
        shared_array[x] = array[x]
    cuda.syncthreads()

    for d in range(0, m):
        if x * 2 ** (d + 1) < n - 1:
            shared_array[x * 2 ** (d + 1) + 2 ** (d + 1) - 1] += shared_array[x * 2 ** (d + 1) + 2 ** d - 1]
        cuda.syncthreads()

    if x == 0:
        shared_array[n - 1] = 0
    cuda.syncthreads()

    for d in range(m - 1, -1, -1):
        if x * 2**(d + 1) < n - 1:
                 temp = shared_array[x * 2 ** (d + 1) + 2**d - 1]
                 shared_array[x * 2**(d + 1) + 2**d - 1] = shared_array[x * 2**(d + 1) + 2**(d + 1) - 1]
                 shared_array[x * 2**(d + 1) + 2**(d + 1) - 1] += temp
        cuda.syncthreads()

    if x < n:
        array[x] = shared_array[x]


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


array = np.array([2, 3, 4, 6, 1, 2, 3], dtype=np.int32)
scanGPU(array)
