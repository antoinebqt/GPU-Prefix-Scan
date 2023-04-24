import math

import numpy as np
from numba import cuda


@cuda.jit
def scanKernel(array, n):
    m = int(math.log2(n))
    #print("wtf")

    for d in range(0, m):
        k = cuda.grid(1)  # Soit k devrait etre en fonction de d
        if k != 1 and k != 3:  # Meilleur if pour vérifier quel thread doit travailler
            print("Par k =", k, "(et d =", d, ") a[", k + pow(2, d + 1) - 1, "] se fait ajouter a[",
                  k + pow(2, d) - 1, "]")
            array[k + pow(2, d + 1) - 1] += array[k + pow(2, d) - 1]  # Peut etre pas les bons index
            cuda.syncthreads()

    #array[n - 1] = 0

    # for d in range(m - 1, -1, -1):
    #    k = cuda.grid(1)
    #    if k != 1 and k != 3:
    #        print("Par k =", k, "(et d =", d, ") t = a[", k + pow(2, d) - 1, "] | a[", k + pow(2, d) - 1, "] = a[",
    #              k + pow(2, d + 1) - 1, "] | a[", k + pow(2, d + 1) - 1, "] += t")
    #        t = array[k + pow(2, d) - 1]
    #        array[k + pow(2, d) - 1] = array[k + pow(2, d + 1) - 1]
    #        array[k + pow(2, d + 1) - 1] += t


def scanGPU(array):
    n = len(array)
    if n > 1024:
        exit(1)

    # Copy array to device
    d_a = cuda.to_device(array)

    # Launch kernel
    threads_per_block = n
    scanKernel[1, threads_per_block](d_a, n)

    # Copy result back to host
    array = d_a.copy_to_host()

    print("Array apres la montée et la descente : ", array)


array = np.array([2, 3, 4, 6], dtype=np.int32)
scanGPU(array)
