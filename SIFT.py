import numpy as np


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
