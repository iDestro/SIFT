import cv2
import numpy as np
from SIFT import SIFT


img = cv2.imread('../dataset/img1.ppm')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sigma = 2
d = int(3*sigma)
img = SIFT.gaussian_blur(img, sigma)

cv2.imshow('ga', img)
cv2.waitKey(0)