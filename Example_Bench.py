from numba import cuda
import numba as nb
import numpy as np
import sys
import math
import Example_Types_Cost
from timeit import default_timer as timer


def benchType(runs, size, type):
    threadsPerBlock = 1024
    blocksPerGrid = math.ceil(size / threadsPerBlock)
    print("Starting", sys._getframe().f_code.co_name)
    print("threadsPerBlock ", threadsPerBlock)
    print("blocksPerGrid", blocksPerGrid)
    result = np.zeros(runs, dtype=np.float32)
    for i in range(runs):
        start = timer()
        Example_Types_Cost.runTypeFloat(blocksPerGrid, threadsPerBlock, size, type)
        cuda.synchronize()
        dt = timer() - start
        print(" ", dt, " s")
        result[i] = dt
    print("Average :", threadsPerBlock, np.average(result[1:]))


if __name__ == '__main__':
    benchType(10, 2000, np.float32)
