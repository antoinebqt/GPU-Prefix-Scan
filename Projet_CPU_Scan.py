import numpy as np


def scanCPU(a):
    n = len(a)
    m = int(np.log2(n))
    print("Array au début : ", a)

    for d in range(0, m):
        for k in range(0, n, pow(2, d + 1)):
            a[k + pow(2, d + 1) - 1] += a[k + pow(2, d) - 1]

    print("Array apres la montée : ", a)

    a[n - 1] = 0
    print("Array apres la mise a 0 : ", a)

    for d in range(m - 1, -1, -1):
        for k in range(0, n, pow(2, d + 1)):
            t = a[k + pow(2, d) - 1]
            a[k + pow(2, d) - 1] = a[k + pow(2, d + 1) - 1]
            a[k + pow(2, d + 1) - 1] += t

    print("Array apres la descente : ", a)


array = np.array([2, 3, 4, 6, 1, 2, 3, 4], dtype=np.int32)
scanCPU(array)
