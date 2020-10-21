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
        cur_sigma = np.power(initial_sigma, octave + i / s)
        size = (int(6*cur_sigma+1), int(6*cur_sigma+1))
        temp_img = SIFT.gaussian_blur(cur_octave_init_img, )
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


# 获取所有极值点
extremal_values_indexes = []
for octave, images in differential_pyramid.items():
    for layer in range(1, len(images) - 1):
        h, w = images[layer].shape
        for i in range(1, h - 1):
            for j in range(1, w - 1):
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


# 除去不好的极值点
valid_extremal = []
img_scale = 1.0 / 255
deriv_scale = img_scale * 0.5
second_deriv_scale = img_scale
cross_deriv_scale = img_scale * 0.25
INT_MAX = 2147483647
SIFT_MAX_INTERP_STEPS = 5
SIFT_IMG_BORDER = 5
contrastThreshold = 0.04
nOctaveLayers = 2

for extremal in extremal_values_indexes:
    octave, layer, i, j = extremal
    up_layer = differential_pyramid[octave][layer + 1].astype(dtype=float)
    cur_layer = differential_pyramid[octave][layer].astype(dtype=float)
    down_layer = differential_pyramid[octave][layer - 1].astype(dtype=float)

    h, w = cur_layer.shape
    xc, xr, xi = 0, 0, 0
    flag = True
    for epoch in range(SIFT_MAX_INTERP_STEPS):
        dD = np.array([
            (cur_layer[i][j + 1] - cur_layer[i][j - 1]) * deriv_scale,
            (cur_layer[i + 1][j] - cur_layer[i - 1][j]) * deriv_scale,
            (up_layer[i][j] - down_layer[i][j]) * deriv_scale
        ])
        v2 = cur_layer[i][j] * 2
        dxx = (cur_layer[i][j + 1] - cur_layer[i][j - 1] - v2) * second_deriv_scale
        dyy = (cur_layer[i + 1][j] - cur_layer[i - 1][j] - v2) * second_deriv_scale
        dss = (up_layer[i][j] - down_layer[i][j] - v2) * second_deriv_scale

        dxy = (cur_layer[i - 1][j - 1] + cur_layer[i + 1][j + 1] - cur_layer[i - 1][j + 1] - cur_layer[i + 1][
            j - 1]) * cross_deriv_scale
        dxs = (down_layer[i][j - 1] + up_layer[i][j + 1] - down_layer[i][j + 1] - up_layer[i][
            j - 1]) * cross_deriv_scale
        dys = (down_layer[i - 1][j] + up_layer[i + 1][j] - down_layer[i + 1][j] - up_layer[i - 1][
            j]) * cross_deriv_scale



        Hessian = np.array([np.array([dxx, dxy, dxs]),
                           np.array([dxy, dyy, dys]),
                           np.array([dxs, dys, dss])]).astype(dtype=float)

        X = np.linalg.pinv(Hessian).dot(dD)
        xi = -X[2]
        xr = -X[1]
        xc = -X[0]

        if np.abs(xi) < 0.5 and np.abs(xr) < 0.5 and np.abs(xc) < 0.5:
            break

        if np.abs(xi) > INT_MAX or np.abs(xr) > INT_MAX or np.abs(xc) > INT_MAX:
            flag = False
            break

        i += np.round(xc)
        j += np.round(xr)
        layer += np.round(xi)
        i = int(i)
        j = int(j)
        layer = int(layer)
        if layer < 0 or layer >= octave_length or i < SIFT_IMG_BORDER or i > h - SIFT_IMG_BORDER or j < SIFT_IMG_BORDER or j > w - SIFT_IMG_BORDER:
            flag = False
            break

        if epoch == SIFT_MAX_INTERP_STEPS - 1:
            break

        up_layer = differential_pyramid[octave][layer + 1]
        cur_layer = differential_pyramid[octave][layer]
        down_layer = differential_pyramid[octave][layer - 1]

        dD = np.array([
            (cur_layer[i][j + 1] - cur_layer[i][j - 1]) * deriv_scale,
            (cur_layer[i + 1][j] - cur_layer[i - 1][j]) * deriv_scale,
            (up_layer[i][j] - down_layer[i][j]) * deriv_scale
        ])

        t = np.sum(dD * np.array([xc, xr, layer]))
        contr = cur_layer[i][j] * img_scale + t * 0.5

        if np.abs(contr) * nOctaveLayers < contrastThreshold:
            flag = False
            break

        v2 = cur_layer[i][j] * 2
        dxy = (cur_layer[i - 1][j - 1] + cur_layer[i + 1][j + 1] - cur_layer[i - 1][j + 1] - cur_layer[i + 1][
            j - 1]) * cross_deriv_scale
        dxs = (down_layer[i][j - 1] + up_layer[i][j + 1] - down_layer[i][j + 1] - up_layer[i][
            j - 1]) * cross_deriv_scale
        dys = (down_layer[i - 1][j] + up_layer[i + 1][j] - down_layer[i + 1][j] - up_layer[i - 1][
            j]) * cross_deriv_scale

        tr = dxx + dyy
        det = dxx * dxx - dxy * dxy
        if tr**2/det < 12.1:
            flag = False
            break

    if flag:
        x = (i+xc)*2**octave
        y = (j+xr)*2**octave
        size = np.power(2.0, (layer+xi)/3)*(1 << octave_length)*2
        valid_extremal.append(np.array([x, y, size]))

print(len(valid_extremal))


#


cv2.waitKey(0)
