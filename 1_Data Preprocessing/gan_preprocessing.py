import os
import cv2
import numpy as np

img = cv2.imread('C:/Users/admin/Desktop/sample1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bright = cv2.subtract(gray, 50)
blur = cv2.GaussianBlur(bright,(15,15),0)
cv2.imwrite('C:/Users/admin/Desktop/blur.jpg',blur)
