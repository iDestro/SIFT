import cv2
import numpy as np
import matplotlib.pyplot as plt
from SIFT import SIFT

img = cv2.imread('../dataset/img1.ppm')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', img)


# laplacian_operator = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
#
# img = SIFT.make_border(img)
# m, n = img.shape
#
# laplacian_img = np.zeros_like(img, dtype=np.float64)
#
# for row in range(1, m-1):
#     for col in range(1, n-1):
#         laplacian_img[row-1:row+2, col-1:col+2] = np.sum(img[row-1:row+2, col-1:col+2]*laplacian_operator)
#
# max_val = np.max(laplacian_img)
# min_val = np.min(laplacian_img)
#
# laplacian_img = (laplacian_img-min_val)/(max_val-min_val)*255
# laplacian_img = np.int8(laplacian_img+0.5)
# laplacian_img = laplacian_img[1:m-1, 1:n-1]

laplacian_img = SIFT.laplacian(img)

print(laplacian_img.shape)
plt.imshow(laplacian_img)
plt.show()
cv2.imshow('laplacian', laplacian_img)
cv2.waitKey(0)