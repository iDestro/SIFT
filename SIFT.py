import numpy as np
import cv2


class SIFT:
    def __int__(self):
        ...

    @staticmethod
    def make_border(img):
        rows, cols = img.shape
        bordered_img = np.zeros([rows+2, cols+2], dtype=np.uint8)
        bordered_img[1:rows+1, 1:cols+1] = img

        for i in range(1, cols+1):
            bordered_img[0, i] = bordered_img[1, i]
            bordered_img[rows+1, i] = bordered_img[rows, i]

        for i in range(1, rows+1):
            bordered_img[i, 0] = bordered_img[i, 1]
            bordered_img[i, cols+1] = bordered_img[i, cols]

        bordered_img[0, 0] = img[0, 0]
        bordered_img[0, cols+1] = img[0, cols-1]
        bordered_img[rows+1, 0] = img[rows-1, 0]
        bordered_img[rows+1, cols+1] = img[rows-1, cols-1]

        return bordered_img

    @staticmethod
    def laplacian(img):
        laplacian_operator = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        img = SIFT.make_border(img)

        m, n = img.shape
        laplacian_img = np.zeros_like(img, dtype=np.float64)

        for row in range(1, m - 1):
            for col in range(1, n - 1):
                laplacian_img[row - 1:row + 2, col - 1:col + 2] = np.sum(
                    img[row - 1:row + 2, col - 1:col + 2] * laplacian_operator)

        max_val = np.max(laplacian_img)
        min_val = np.min(laplacian_img)

        laplacian_img = (laplacian_img - min_val) / (max_val - min_val) * 255
        laplacian_img = np.int8(laplacian_img + 0.5)
        laplacian_img = laplacian_img[1:m - 1, 1:n - 1]
        return laplacian_img

    @staticmethod
    def gaussian_blur(img, sigma):

        m, n = img.shape
        d = int(sigma * 3)
        # gaussian_filter = np.zeros([2 * d + 1, 2 * d + 1], dtype=np.float)
        # padded_img = np.zeros([m + 2 * d, n + 2 * d], dtype=np.float)
        # padded_img[d:m + d, d:n + d] = img
        # filtered_img = np.zeros_like(padded_img, dtype=np.float)
        #
        # for i in range(2 * d + 1):
        #     for j in range(2 * d + 1):
        #         gaussian_filter[i, j] = 1.0 / (2 * np.pi * sigma ** 2) * np.exp(
        #             -((i - d) ** 2 + (j - d) ** 2) / (2 * sigma ** 2))
        #
        # for i in range(d, m + d):
        #     for j in range(d, n + d):
        #         filtered_img[i, j] = np.sum(padded_img[i - d:i + d + 1, j - d:j + d + 1] * gaussian_filter)
        #
        # filtered_img = filtered_img[d:m + d, d:n + d]
        # max_val = np.max(filtered_img)
        # min_val = np.min(filtered_img)
        # filtered_img = (filtered_img - min_val) / (max_val - min_val) * 255
        # filtered_img = np.uint8(filtered_img + 0.5)

        filtered_img = cv2.GaussianBlur(img, (2*d+1, 2*d+1), sigma)
        return filtered_img
