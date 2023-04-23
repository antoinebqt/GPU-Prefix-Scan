from PIL import Image

with Image.open("Lab_2_Img.jpg") as im:
    im.rotate(45).save("Lab_2_Img_Result.jpg", "JPEG")
    print("Size: ", im.size)