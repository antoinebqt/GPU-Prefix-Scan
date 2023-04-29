import math
import sys
import numpy as np
from numba import cuda


# Kernel for single thread block
@cuda.jit()
def scanKernel(array, n):
    m = round(math.log2(n))
    global_id = cuda.grid(1)

    # Creation of the shared array and copy from the array
    shared_array = cuda.shared.array(shape=1024, dtype=np.int32)
    if global_id < n:
        shared_array[global_id] = array[global_id]
    cuda.syncthreads()

    # Up phase
    for d in range(0, m):

        # Compute k from the global id and the step
        k = global_id * pow(2, d + 1)

        if k < n - 1:  # Check if the thread should work
            shared_array[k + pow(2, d + 1) - 1] += shared_array[k + pow(2, d) - 1]
        cuda.syncthreads()

    # Only one thread set the last element to 0
    if global_id == 0:
        shared_array[n - 1] = 0
    cuda.syncthreads()

    # Down phase
    for d in range(m - 1, -1, -1):
        # Compute k from the global id and the step
        k = global_id * pow(2, d + 1)

        if k < n - 1:  # Check if the thread should work
            t = shared_array[k + pow(2, d) - 1]
            shared_array[k + pow(2, d) - 1] = shared_array[k + pow(2, d + 1) - 1]
            shared_array[k + pow(2, d + 1) - 1] += t
        cuda.syncthreads()

    # Copy the result back to the array
    array[global_id] = shared_array[global_id]


# Kernel for multiple thread blocks
@cuda.jit
def scanKernel2(array, arrayS, n, threads_per_block):
    m = round(math.log2(threads_per_block))
    local_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    global_id = cuda.grid(1)

    # Creation of the shared array and copy from the array
    shared_array = cuda.shared.array(shape=1024, dtype=np.int32)
    if global_id < n:
        shared_array[local_id] = array[global_id]
    cuda.syncthreads()

    # Up phase
    for d in range(0, m):
        # Compute k from the local id and the step
        k = local_id * pow(2, d + 1)

        if k < n - 1:  # Check if the thread should work
            shared_array[k + pow(2, d + 1) - 1] += shared_array[k + pow(2, d) - 1]
        cuda.syncthreads()

    # Save the last element of the shared array in the sum array
    arrayS[block_id] = shared_array[threads_per_block - 1]

    # Only one thread set the last element to 0
    if local_id == 0:
        shared_array[threads_per_block - 1] = 0
    cuda.syncthreads()

    # Down phase
    for d in range(m - 1, -1, -1):
        # Compute k from the local id and the step
        k = local_id * pow(2, d + 1)

        if k < n - 1:  # Check if the thread should work
            t = shared_array[k + pow(2, d) - 1]
            shared_array[k + pow(2, d) - 1] = shared_array[k + pow(2, d + 1) - 1]
            shared_array[k + pow(2, d + 1) - 1] += t
        cuda.syncthreads()

    # Copy the result back to the array
    array[global_id] = shared_array[local_id]


# Kernel for adding the prefix sum array to the array
@cuda.jit
def addPrefixSum(array, arrayPrefixSum, n):
    global_id = cuda.grid(1)
    block_id = cuda.blockIdx.x

    if global_id < n:  # Check if the thread should work
        array[global_id] += arrayPrefixSum[block_id]


# Kernel for adding the initial array to the array
@cuda.jit
def addInitialArray(array, initialArray, n):
    global_id = cuda.grid(1)

    if global_id < n:  # Check if the thread should work
        array[global_id] += initialArray[global_id]


# Function to launch the prefix scan algorithm on the GPU
def scanGPU(array, THREADS_PER_BLOCK, INDEPENDENT):
    n = len(array)

    if n <= THREADS_PER_BLOCK:  # If the array can be processed by a single thread block
        m = int(math.pow(2, math.ceil(math.log2(n))))  # Compute the next power of 2

        # Create a new array of size m with the original array followed by zeros
        padded_array = np.zeros(m, dtype=np.int32)
        padded_array[:n] = array

        # Copy array to device
        d_a = cuda.to_device(padded_array)

        # Compute the number of threads per thread block
        threads_per_block = m

        # Launch kernel for scan prefix
        scanKernel[1, threads_per_block](d_a, m)

        # Copy result back to host
        array = d_a.copy_to_host()[:n]
    else:  # If the array cannot be processed by a single thread block
        # Compute the number of thread blocks and the number of threads per thread block
        threads_per_block = THREADS_PER_BLOCK
        blocks_on_grid = int(math.ceil(n / threads_per_block))

        # Copy array to device
        d_a = cuda.to_device(array)

        # Create the sum array and copy it to device
        d_aS = cuda.to_device(np.zeros(blocks_on_grid, dtype=np.int32))

        # Launch kernel for scan prefix
        scanKernel2[blocks_on_grid, threads_per_block](d_a, d_aS, n, threads_per_block)

        if not INDEPENDENT:
            # Copy the sum array back to host
            arrayS = d_aS.copy_to_host()

            # Compute the scan prefix of the sum array
            prefixSum = scanGPU(arrayS, THREADS_PER_BLOCK, INDEPENDENT)

            # Copy the prefix sum array to device
            d_apS = cuda.to_device(prefixSum)

            # Launch kernel for adding the prefix sum array to the result array
            addPrefixSum[blocks_on_grid, threads_per_block](d_a, d_apS, n)

        # Copy result back to host
        array = d_a.copy_to_host()

    return array


# Function to launch the prefix scan algorithm on the GPU and add the initial array if needed
def scanGPU2(array, THREADS_PER_BLOCK, INDEPENDENT, INCLUSIVE):
    n = len(array)

    # Launch the prefix scan algorithm and get the result
    result = scanGPU(array, THREADS_PER_BLOCK, INDEPENDENT)

    if INCLUSIVE:
        # Copy result and initial array to device
        d_aR = cuda.to_device(result)
        d_aI = cuda.to_device(array)

        # Compute the number of thread blocks and the number of threads per thread block
        threads_per_block = THREADS_PER_BLOCK
        blocks_on_grid = int(math.ceil(n / threads_per_block))

        # Launch kernel for adding the initial array to the result array
        addInitialArray[blocks_on_grid, threads_per_block](d_aR, d_aI, n)

        # Copy result back to host
        result = d_aR.copy_to_host()

    return result


# Main function
def main():
    # Default options values
    THREADS_PER_BLOCK = 1024
    INDEPENDENT = False
    INCLUSIVE = False

    # Read the input file and the options
    input_file = sys.argv[1]

    for i in range(2, len(sys.argv)):
        if sys.argv[i] == "--tb":
            THREADS_PER_BLOCK = int(sys.argv[i + 1])
        if sys.argv[i] == "--independent":
            INDEPENDENT = True
        if sys.argv[i] == "--inclusive":
            INCLUSIVE = True

    # Get the input array from the file
    with open(input_file, 'r') as f:
        array = [int(x) for x in f.read().split(',')]

    # Perform the prefix scan on the GPU
    result = scanGPU2(array, THREADS_PER_BLOCK, INDEPENDENT, INCLUSIVE)

    # Print the result
    print(','.join(str(x) for x in result))


if __name__ == '__main__':
    main()
