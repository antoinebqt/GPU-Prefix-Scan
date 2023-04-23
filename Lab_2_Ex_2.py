from PIL import Image
import numpy as np

with Image.open("Lab_2_Img.jpg") as im:
    w, h = im.size

    print("Width: ", w)
    print("Height: ", h)

    imArray = np.asarray(im)
    imArray = imArray.transpose(1, 0, 2)
    imArray = np.ascontiguousarray(imArray)

    print("Dimension of array: ", imArray.shape)