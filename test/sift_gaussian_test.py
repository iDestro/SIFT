from SIFT import SIFT
import cv2

img = cv2.imread('../dataset/img1.ppm')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = SIFT.gaussian_blur(img, 10)
cv2.imshow('gaussian img', img)
cv2.waitKey(0)
