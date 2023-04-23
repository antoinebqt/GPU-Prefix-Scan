from numba import cuda


@cuda.jit
def coordinates2D():
    local_idX = cuda.threadIdx.x
    local_idY = cuda.threadIdx.y

    global_idX, global_idY = cuda.grid(2)

    # Print local and global IDs
    print("(", local_idX, ",", local_idY, ") in ", "(", cuda.blockIdx.x, ",", cuda.blockIdx.y, ") -> (", global_idX,
          ",", global_idY, ")")


gridSize = (2, 2, 1)
blockSize = (4, 2, 1)
coordinates2D[gridSize, blockSize]()
cuda.synchronize()  # Wait for kernel to finish execution
