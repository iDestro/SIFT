import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../dataset/img1.ppm')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('origin_img', img)
m, n = img.shape

sigma = 10
d = int(sigma*3)
gaussian_filter = np.zeros([2*d+1, 2*d+1], dtype=np.float)

padded_img = np.zeros([m+2*d, n+2*d], dtype=np.float)

padded_img[d:m+d, d:n+d] = img

filtered_img = np.zeros_like(padded_img, dtype=np.float)

for i in range(2*d+1):
    for j in range(2*d+1):
        gaussian_filter[i, j] = 1.0/(2*np.pi*sigma**2)*np.exp(-((i-d)**2+(j-d)**2)/(2*sigma**2))

print(np.sum(gaussian_filter))
plt.imshow(gaussian_filter)
plt.show()
for i in range(d, m+d):
    for j in range(d, n+d):
        print(np.sum(padded_img[i-d:i+d+1, j-d:j+d+1]*gaussian_filter))
        filtered_img[i, j] = np.sum(padded_img[i-d:i+d+1, j-d:j+d+1]*gaussian_filter)
filtered_img = filtered_img[d:m+d, d:n+d]
print(filtered_img)
max_val = np.max(filtered_img)
print(max_val)
min_val = np.min(filtered_img)
print(min_val)
filtered_img = (filtered_img-min_val)/(max_val-min_val)*255
filtered_img = filtered_img+0.5
filtered_img = np.uint8(filtered_img+0.5)
print(filtered_img)
cv2.imshow('gaussian_filtered_img', filtered_img)
cv2.waitKey(0)