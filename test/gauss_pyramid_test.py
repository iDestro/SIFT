import cv2
from SIFT import SIFT
import numpy as np

img = cv2.imread('../dataset/img1.ppm')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
w, h = img.shape
octave_length = int(np.log(min(w, h))) - 3
s = 5
print('octave_length: ', octave_length)

initial_sigma = 1.6

# cv2.imshow('origin', img)
# img = cv2.resize(img, (h//2, w//2))
# cv2.imshow('resize', img)

octaves = {i: [] for i in range(octave_length)}

for octave in range(octave_length):
    downsample_size = int(np.power(2, octave))
    if octave == 0:
        cur_octave_init_img = img
    else:
        cur_octave_init_img = octaves[octave - 1][2]
        w, h = cur_octave_init_img.shape
        cur_octave_init_img = cv2.resize(cur_octave_init_img, (h // 2, w // 2))
    for i in range(s):
        temp_img = SIFT.gaussian_blur(cur_octave_init_img, sigma=np.power(initial_sigma, octave + i / s))
        octaves[octave].append(temp_img)

# for key, val in octaves.items():
#     for i, img in enumerate(val):
#         cv2.imshow(str(key)+":"+str(i), img)

differential_pyramid = {i: [] for i in range(octave_length)}

for key, val in octaves.items():
    for i in range(0, s - 1):
        differential_pyramid[key].append(val[i + 1] - val[i])

# for key, val in differential_pyramid.items():
#     for i, img in enumerate(val):
#         print(img)
#         cv2.imshow(str(key)+":"+str(i), img)


extremal_values_indexes = []
for octave, images in differential_pyramid.items():
    for layer in range(1, len(images) - 1):
        h, w = images[layer].shape
        for i in range(1, h-1):
            for j in range(1, w-1):
                cur_pixel = images[layer][i][j]
                if (
                        cur_pixel > images[layer][i][j - 1] and cur_pixel > images[layer][i][j + 1] and
                        cur_pixel > images[layer][i - 1][j] and cur_pixel > images[layer][i + 1][j] and
                        cur_pixel > images[layer][i - 1][j - 1] and cur_pixel > images[layer][i + 1][j + 1] and
                        cur_pixel > images[layer][i - 1][j + 1] and cur_pixel > images[layer][i + 1][j - 1] and

                        cur_pixel > images[layer - 1][i][j] and
                        cur_pixel > images[layer - 1][i][j - 1] and cur_pixel > images[layer - 1][i][j + 1] and
                        cur_pixel > images[layer - 1][i - 1][j] and cur_pixel > images[layer - 1][i + 1][j] and
                        cur_pixel > images[layer - 1][i - 1][j - 1] and cur_pixel > images[layer - 1][i + 1][j + 1] and
                        cur_pixel > images[layer - 1][i - 1][j + 1] and cur_pixel > images[layer - 1][i + 1][j - 1] and

                        cur_pixel > images[layer + 1][i][j] and
                        cur_pixel > images[layer + 1][i][j - 1] and cur_pixel > images[layer + 1][i][j + 1] and
                        cur_pixel > images[layer + 1][i - 1][j] and cur_pixel > images[layer + 1][i + 1][j] and
                        cur_pixel > images[layer + 1][i - 1][j - 1] and cur_pixel > images[layer + 1][i + 1][j + 1] and
                        cur_pixel > images[layer + 1][i - 1][j + 1] and cur_pixel > images[layer + 1][i + 1][j - 1]
                ) or (
                        cur_pixel < images[layer][i][j - 1] and cur_pixel < images[layer][i][j + 1] and
                        cur_pixel < images[layer][i - 1][j] and cur_pixel < images[layer][i + 1][j] and
                        cur_pixel < images[layer][i - 1][j - 1] and cur_pixel < images[layer][i + 1][j + 1] and
                        cur_pixel < images[layer][i - 1][j + 1] and cur_pixel < images[layer][i + 1][j - 1] and

                        cur_pixel < images[layer - 1][i][j] and
                        cur_pixel < images[layer - 1][i][j - 1] and cur_pixel < images[layer - 1][i][j + 1] and
                        cur_pixel < images[layer - 1][i - 1][j] and cur_pixel < images[layer - 1][i + 1][j] and
                        cur_pixel < images[layer - 1][i - 1][j - 1] and cur_pixel < images[layer - 1][i + 1][j + 1] and
                        cur_pixel < images[layer - 1][i - 1][j + 1] and cur_pixel < images[layer - 1][i + 1][j - 1] and

                        cur_pixel < images[layer + 1][i][j] and
                        cur_pixel < images[layer + 1][i][j - 1] and cur_pixel < images[layer + 1][i][j + 1] and
                        cur_pixel < images[layer + 1][i - 1][j] and cur_pixel < images[layer + 1][i + 1][j] and
                        cur_pixel < images[layer + 1][i - 1][j - 1] and cur_pixel < images[layer + 1][i + 1][j + 1] and
                        cur_pixel < images[layer + 1][i - 1][j + 1] and cur_pixel < images[layer + 1][i + 1][j - 1]
                ):
                    extremal_values_indexes.append((octave, layer, i, j))

for i in extremal_values_indexes:
    print(i)

print(len(extremal_values_indexes))
cv2.waitKey(0)
