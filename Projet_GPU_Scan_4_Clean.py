import math
import sys

import numpy as np
from numba import cuda

@cuda.jit
def scanKernel(array, n):
    m = round(math.log2(n))
    idx = cuda.grid(1)

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

            t = shared_array[k + pow(2, d) - 1]
            shared_array[k + pow(2, d) - 1] = shared_array[k + pow(2, d + 1) - 1]
            shared_array[k + pow(2, d + 1) - 1] += t
        cuda.syncthreads()

    # Copie du résultat dans l'array
    array[global_id] = shared_array[local_id]


@cuda.jit
def addPrefixSum(array, arrayPrefixSum, n):
    global_id = cuda.grid(1)
    block_id = cuda.blockIdx.x
    if global_id < n:
        array[global_id] += arrayPrefixSum[block_id]


@cuda.jit
def addInitialArray(array, initialArray, n):
    global_id = cuda.grid(1)
    if global_id < n:
        array[global_id] += initialArray[global_id]


def scanGPU(array, THREADS_PER_BLOCK, INDEPENDENT):
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

        prefixSum = scanGPU(arrayS, THREADS_PER_BLOCK, INDEPENDENT)
        d_apS = cuda.to_device(prefixSum)

        if not INDEPENDENT:
            addPrefixSum[blocks_on_grid, threads_per_block](d_a, d_apS, n)

        array = d_a.copy_to_host()

    return array


def scanGPU2(array, THREADS_PER_BLOCK, INDEPENDENT, INCLUSIVE):
    n = len(array)
    result = scanGPU(array, THREADS_PER_BLOCK, INDEPENDENT)

    if INCLUSIVE:
        d_aR = cuda.to_device(result)
        d_aI = cuda.to_device(array)

        threads_per_block = THREADS_PER_BLOCK
        blocks_on_grid = int(math.ceil(n / threads_per_block))

        addInitialArray[blocks_on_grid, threads_per_block](d_aR, d_aI, n)

        result = d_aR.copy_to_host()

    return result


def main():
    input_file = sys.argv[1]
    THREADS_PER_BLOCK = 1024
    INDEPENDENT = False
    INCLUSIVE = False

    for i in range(2, len(sys.argv)):
        if sys.argv[i] == "--tb":
            THREADS_PER_BLOCK = int(sys.argv[i + 1])
        if sys.argv[i] == "--independent":
            INDEPENDENT = True
        if sys.argv[i] == "--inclusive":
            INCLUSIVE = True

    # Read the input array from the file
    with open(input_file, 'r') as f:
        array = [int(x) for x in f.read().split(',')]

    # Perform the scan on the GPU
    result = scanGPU2(array, THREADS_PER_BLOCK, INDEPENDENT, INCLUSIVE)

    # Print the result
    print(','.join(str(x) for x in result))


if __name__ == '__main__':
    main()
