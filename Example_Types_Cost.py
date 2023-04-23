from numba import cuda
import numba as nb
import numpy as np
import sys
import math


@cuda.jit
def intensiveKernel(array):
    local_id = cuda.threadIdx.x
    global_id = cuda.grid(1)
    if global_id < array.shape[0]:
        array[global_id] = array[global_id] * array[global_id]


def runTypeInt(blocksPerGrid, threadsPerBlock, size, type):
    A = np.random.randint(1, 100, size=size, dtype=type)
    d_A = cuda.to_device(A)
    # Execute kernel
    intensiveKernel[blocksPerGrid, threadsPerBlock](d_A)
    cuda.synchronize()
    # Copy back the modified array
    A = d_A.copy_to_host()


def runTypeFloat(blocksPerGrid, threadsPerBlock, size, type):
    A = np.random.rand(size).astype(type)
    d_A = cuda.to_device(A)
    # Execute kernel
    intensiveKernel[blocksPerGrid, threadsPerBlock](d_A)
    cuda.synchronize()
    # Copy back the modified array
    A = d_A.copy_to_host()


def runAll(size):
    ty = [np.uint8, np.uint16, np.uint32]
    threadsPerBlock = 1024
    blocksPerGrid = math.ceil(size / threadsPerBlock)
    print("Starting", sys._getframe().f_code.co_name)
    print("threadsPerBlock ", threadsPerBlock)
    print("blocksPerGrid", blocksPerGrid)
    for t in ty:
        print("\t Memory size", t.__name__, np.dtype(t).itemsize * size // 1024 // 1024, "MB")
        print("\t Executing kernel")
        runTypeInt(blocksPerGrid, threadsPerBlock, size, t)

    ty = [np.float32, np.float64]
    for t in ty:
        print("\t Memory size", t.__name__, np.dtype(t).itemsize * size // 1024 // 1024, "MB")
        print("\t Executing kernel")
        runTypeFloat(blocksPerGrid, threadsPerBlock, size, t)


# kernel_gpu=cuda.jit(intensiveKernel)
# print(type(intensiveKernel), " compiled to ", type(kernel_gpu))
# print(kernel_gpu.inspect_types())

# object_methods = [method_name for method_name in dir(kernel_gpu)
#                   if callable(getattr(object, method_name))]
# for d in object_methods:
#     print(d)


if __name__ == '__main__':
    runAll(2 ** 10)
