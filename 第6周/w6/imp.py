from PIL import Image
from PIL import ImageFilter
import numpy as np
import matplotlib.pyplot as plt


im = Image.open('lx.jpg')
print(im.format, im.size, im.mode)#size width, height
im.show()

array = np.asarray(im)
print(array.shape)
#print(array)

box = (10, 10, 250, 250)
region = im.crop(box)
region.show()

size = (128, 128)
resize = im.resize((128, 128))
resize.show()

angle = 45 #逆时针
rotate = im.rotate(angle)
rotate.show()

blur = im.filter(ImageFilter.BLUR)
blur.show()

sharpen = im.filter(ImageFilter.SHARPEN)
sharpen.show()

eem = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
eem.show()

con = im.filter(ImageFilter.CONTOUR)
con.show()

plt.figure()

plt.subplot(2,2,1)
plt.imshow(blur)
plt.subplot(2,2,2)
plt.imshow(sharpen)
plt.subplot(2,2,3)
plt.imshow(eem)
plt.subplot(2,2,4)
plt.imshow(con)

plt.show()

con.save('lx_contour.png')