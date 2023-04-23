from numba import cuda
import numpy as np


@cuda.jit
def writeGlobalID(array):
    global_id = cuda.grid(1)
    array[global_id] = global_id


threadsPerBlock = 16
blocksPerGrid = 1

# Generate array with size matching total number of threads
A = np.zeros(threadsPerBlock * blocksPerGrid, dtype=np.uint16)
# Send array to device
d_A = cuda.to_device(A)

writeGlobalID[blocksPerGrid, threadsPerBlock](d_A)
cuda.synchronize()

A = d_A.copy_to_host()
# Print a subsequence to check
print(A)