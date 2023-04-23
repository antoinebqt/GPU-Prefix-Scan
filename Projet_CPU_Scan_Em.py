import math
import numpy as np


def scanCPU(array):
    n = array.size
    m = int(math.log2(n))

    print(array)

    # Monté
    for d in range(0, m):
        print("d: ", d)

        for k in range(0, n, 2 ** (d + 1)):
            print("k: ", k)

            array[k + 2 ** (d + 1) - 1] += array[k + 2 ** d - 1]

    print(array)

    # Descente
    array[n - 1] = 0
    for d in range(m - 1, -1, -1):
        print("d: ", d)

        for k in range(0, n, 2 ** (d + 1)):
            print("k: ", k)

            temp = array[k + 2 ** d - 1]
            array[k + 2 ** d - 1] = array[k + 2 ** (d + 1) - 1]
            array[k + 2 ** (d + 1) - 1] += temp

    print(array)


def generate_random_array(size):
    random_array = np.random.randint(0, 21, size=size)
    return random_array


arrayTD = np.array([2, 3, 4, 6])

scanCPU(arrayTD)
# scanCPU(generate_random_array(64))