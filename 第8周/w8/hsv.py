import sys
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread(sys.argv[1])
height, width, channels = img.shape
print(f'{height},{width},{channels}')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.subplot(2,2,1)
plt.imshow(gray, cmap='gray')

plt.subplot(2,2,2)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
plt.imshow(hsv)

avgh, avgs, avgv = cv.mean(img)[:-1]

avgh /= 3
avgs /= 3
avgv /= 3

print(avgh,avgs,avgv)

plt.subplot(2,2,3)
th = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
plt.imshow(th,cmap = 'gray')

plt.subplot(2,2,4)
edges = cv.Canny(gray,70,100)
plt.imshow(edges,cmap = 'gray')
plt.show()


