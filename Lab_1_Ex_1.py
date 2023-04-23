from numba import cuda


@cuda.jit
def firstKernel():
    print("hello")


gridSize = 1
blockSize = 4
firstKernel[gridSize, blockSize]()
cuda.synchronize()  # Wait for kernel to finish execution
