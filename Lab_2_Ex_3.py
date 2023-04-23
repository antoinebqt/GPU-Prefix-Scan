from PIL import Image
import numpy as np
from numba import cuda


# Given a thread block of size (16,16,1) and an image, compute the grid size needed to process it.
# 256
# 1280720 = 921600
# 921600/256
# 3600 block par grid

def getImageAsArray():
    with Image.open("image.jpg") as im:
        w, h = im.size

        print('width: ', w)
        print('height:', h)
        a = np.array(im)
        return a


@cuda.jit
def convertToBW(imageArray):
    x, y = cuda.grid(2)
    if x < imageArray.shape[0] and y < imageArray.shape[1]:
        colorVal = (0.3 * imageArray[x][y][0]) + (0.59 * imageArray[x][y][1]) + (
                0.11 * imageArray[x][y][2])

        imageArray[x][y][0] = colorVal
        imageArray[x][y][1] = colorVal
        imageArray[x][y][2] = colorVal


blocksPerGrid = (51,51)
threadsPerBlock = (16, 16, 1)

imgArray = getImageAsArray()
d_A = cuda.to_device(imgArray)

convertToBW[blocksPerGrid, threadsPerBlock](d_A)

cuda.synchronize()
# Copy back the modified array
A = d_A.copy_to_host()

new_im = Image.fromarray(A)
new_im.save('TestGPU.jpg')