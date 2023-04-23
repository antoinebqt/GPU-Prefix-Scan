from numba import cuda


@cuda.jit
def coordinates1D():
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x

    # Calculate global and local thread IDs
    global_id = tx + bx * bw
    local_id = tx

    # Print local and global IDs
    print("Thread ", local_id, "in block ", bx, " has global ID ", global_id)


gridSize = 2
blockSize = 8
coordinates1D[gridSize, blockSize]()
cuda.synchronize()  # Wait for kernel to finish execution
