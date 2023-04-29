import numpy as np


def scanCPU(a):
    n = len(a)
    m = int(np.log2(n))
    #print("Array au début : ", a)

    for d in range(0, m):
        for k in range(0, n, pow(2, d + 1)):
            a[k + pow(2, d + 1) - 1] += a[k + pow(2, d) - 1]

    #print("Array apres la montée : ", a)

    a[n - 1] = 0
    #print("Array apres la mise a 0 : ", a)

    for d in range(m - 1, -1, -1):
        for k in range(0, n, pow(2, d + 1)):
            t = a[k + pow(2, d) - 1]
            a[k + pow(2, d) - 1] = a[k + pow(2, d + 1) - 1]
            a[k + pow(2, d + 1) - 1] += t

    print(a)


array = np.array(
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
    dtype=np.int32)
scanCPU(array)
