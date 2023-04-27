import math

import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 2
verbose = False


@cuda.jit
def scanKernel(array, n):
    m = round(math.log2(n))
    idx = cuda.grid(1)

    if idx == 0 and verbose:
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


@cuda.jit
def scanKernel2(array, arrayS, n, threads_per_block):
    m = round(math.log2(threads_per_block))
    local_id = cuda.threadIdx.x
    global_id = cuda.grid(1)
    block_id = cuda.blockIdx.x

    if cuda.grid(1) == 0 and verbose:
        print("n :", n)
        print("m :", m)
        print("threads_per_block :", threads_per_block)

    # Création de l'array partagé et copie depuis l'array
    shared_array = cuda.shared.array(shape=1024, dtype=np.int32)
    if global_id < n:
        shared_array[local_id] = array[global_id]
    cuda.syncthreads()

    # Montée
    for d in range(0, m):
        k = local_id * pow(2, d + 1)
        if k < n - 1:
            shared_array[k + pow(2, d + 1) - 1] += shared_array[k + pow(2, d) - 1]
        cuda.syncthreads()

    # Ajout de la somme dans l'array somme
    arrayS[block_id] = shared_array[threads_per_block - 1]

    # Mise a 0 du dernier élément
    if local_id == 0:
        shared_array[threads_per_block - 1] = 0
    cuda.syncthreads()

    # Descente
    for d in range(m - 1, -1, -1):
        k = local_id * pow(2, d + 1)
        if k < n - 1:
            if verbose:
                print("(Bloc", block_id, ") Par k =", k, "(et d =", d, ") t = sa[", k + pow(2, d) - 1, "](",
                      shared_array[k + pow(2, d) - 1], ") | sa[", k + pow(2, d) - 1, "](",
                      shared_array[k + pow(2, d) - 1], ") = sa[", k + pow(2, d + 1) - 1, "](",
                      shared_array[k + pow(2, d + 1) - 1], ") | sa[", k + pow(2, d + 1) - 1, "](",
                      shared_array[k + pow(2, d + 1) - 1], ") += t")
            t = shared_array[k + pow(2, d) - 1]
            shared_array[k + pow(2, d) - 1] = shared_array[k + pow(2, d + 1) - 1]
            shared_array[k + pow(2, d + 1) - 1] += t
        cuda.syncthreads()

    # Copie du résultat dans l'array
    array[global_id] = shared_array[local_id]


@cuda.jit
def addPrefixSum(array, arrayPrefixSum):
    global_id = cuda.grid(1)
    block_id = cuda.blockIdx.x
    array[global_id] += arrayPrefixSum[block_id]


def scanGPU(array):
    n = len(array)
    if n <= THREADS_PER_BLOCK:
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
    else:
        # Copy array to device
        d_a = cuda.to_device(array)

        threads_per_block = THREADS_PER_BLOCK
        blocks_on_grid = int(math.ceil(n / threads_per_block))

        d_aS = cuda.to_device(np.zeros(blocks_on_grid, dtype=np.int32))

        scanKernel2[blocks_on_grid, threads_per_block](d_a, d_aS, n, threads_per_block)

        arrayS = d_aS.copy_to_host()
        if verbose: print("arrayS :", arrayS)

        prefixSum = scanGPU(arrayS)
        d_apS = cuda.to_device(prefixSum)
        addPrefixSum[blocks_on_grid, threads_per_block](d_a, d_apS)

        array = d_a.copy_to_host()

    return array


array = np.array([2, 9, 15, 13, 10, 20, 2, 3, 1, 0], dtype=np.int32)
# array = np.random.randint(low=1, high=100, size=1024, dtype='int32')

print("Array apres la montée et la descente : ", scanGPU(array))
