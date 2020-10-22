import numpy as np
from KeyPoint import KeyPoint
import cv2

FLT_EPSILON = 1.192092896e-07
SIFT_INTVLS = 3
# 最初的高斯模糊的尺度默认值，即假设的第0组（某些地方叫做-1组）的尺度
SIFT_SIGMA = 1.6
# 关键点对比 |D(x)|的默认值，公式14的阈值
SIFT_CONTR_THR = 0.04
# 关键点的主曲率比的默认阈值
SIFT_CURV_THR = 10.
# 是否在构建高斯金字塔之前扩展图像的宽和高位原来的两倍（即是否建立-1组）
SIFT_IMG_DBL = True
# 描述子直方图数组的默认宽度，即描述子建立中的4*4的周边区域
SIFT_DESCR_WIDTH = 4
# 每个描述子数组中的默认柱的个数（ 4*4*8=128）
SIFT_DESCR_HIST_BINS = 8
# 假设输入图像的高斯模糊的尺度
SIFT_INIT_SIGMA = 0.5

# width of border in which to ignore keypoints
SIFT_IMG_BORDER = 5
# 公式12的为了寻找关键点插值中心的最大迭代次数
SIFT_MAX_INTERP_STEPS = 5

# 方向梯度直方图中的柱的个数
SIFT_ORI_HIST_BINS = 36

# determines gaussian sigma for orientation assignment
SIFT_ORI_SIG_FCTR = 1.5

# determines the radius of the region used in orientation assignment
SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR

# orientation magnitude relative to max that results in new feature
SIFT_ORI_PEAK_RATIO = 0.8

# determines the size of a single descriptor orientation histogram
SIFT_DESCR_SCL_FCTR = 3.

# threshold on magnitude of elements of descriptor vector
SIFT_DESCR_MAG_THR = 0.2

# factor used to convert floating-point descriptor to unsigned char
SIFT_INT_DESCR_FCTR = 512.

# intermediate type used for DoG pyramids
SIFT_FIXPT_SCALE = 1

INT_MAX = 2147483647


class SIFT:
    def __init__(self, n_octave_layers=3, contrast_threshold=0.04, edge_threshold=10, sigma=1.6):
        self.n_octave_layers = n_octave_layers
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        self.key_points = []
        self.descriptors = []
        self.img = None
        self.n_octaves = None
        self.gaussian_pyramid = None
        self.dog_pyramid = None

    def create_initial_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape
        self.img = img
        self.n_octaves = int(np.log(min(h, w)))-3
        self.gaussian_pyramid = {i: [] for i in range(self.n_octaves)}
        self.dog_pyramid = {i: [] for i in range(self.n_octaves)}
        # self.build_gaussian_pyramid(img)
        # self.build_dog_pyramid()
        # self.find_scale_space_extrema()
        # self.calc_descriptors()

    def build_gaussian_pyramid(self):
        for octave in range(self.n_octaves):
            if octave == 0:
                cur_octave_init_img = self.img
            else:
                cur_octave_init_img = self.gaussian_pyramid[octave - 1][2]
                w, h = cur_octave_init_img.shape
                cur_octave_init_img = cv2.resize(cur_octave_init_img, (h // 2, w // 2))
            for i in range(self.n_octave_layers + 3):
                cur_sigma = np.power(self.sigma, octave + i / (self.n_octave_layers + 3))
                size = (int(6 * cur_sigma + 1), int(6 * cur_sigma + 1))
                if int(6 * cur_sigma + 1) % 2 == 0:
                    size = (int(6 * cur_sigma + 2), int(6 * cur_sigma + 2))
                temp_img = cv2.GaussianBlur(cur_octave_init_img, ksize=size, sigmaX=cur_sigma, sigmaY=cur_sigma)
                self.gaussian_pyramid[octave].append(temp_img)

    def build_dog_pyramid(self):
        for octave, layer in self.gaussian_pyramid.items():
            for i in range(0, self.n_octave_layers + 2):
                self.dog_pyramid[octave].append(layer[i + 1] - layer[i])

    def find_scale_space_extrema(self):
        threshold = int(np.round(0.5 * self.contrast_threshold / self.n_octave_layers * 255))
        n = SIFT_ORI_HIST_BINS
        for octave, images in self.dog_pyramid.items():
            for layer in range(1, len(images) - 1):
                h, w = images[layer].shape
                for i in range(1, h - 1):
                    for j in range(1, w - 1):
                        cur_pixel = images[layer][i][j]
                        if (np.abs(cur_pixel) > threshold and
                                ((cur_pixel > 0 and
                                  cur_pixel > images[layer][i][j - 1] and
                                  cur_pixel > images[layer][i][j + 1] and
                                  cur_pixel > images[layer][i - 1][j] and
                                  cur_pixel > images[layer][i + 1][j] and
                                  cur_pixel > images[layer][i - 1][j - 1] and
                                  cur_pixel > images[layer][i + 1][j + 1] and
                                  cur_pixel > images[layer][i - 1][j + 1] and
                                  cur_pixel > images[layer][i + 1][j - 1] and
                                  cur_pixel > images[layer - 1][i][j] and
                                  cur_pixel > images[layer - 1][i][j - 1] and
                                  cur_pixel > images[layer - 1][i][j + 1] and
                                  cur_pixel > images[layer - 1][i - 1][j] and
                                  cur_pixel > images[layer - 1][i + 1][j] and
                                  cur_pixel > images[layer - 1][i - 1][j - 1] and
                                  cur_pixel > images[layer - 1][i + 1][j + 1] and
                                  cur_pixel > images[layer - 1][i - 1][j + 1] and
                                  cur_pixel > images[layer - 1][i + 1][j - 1] and
                                  cur_pixel > images[layer + 1][i][j] and
                                  cur_pixel > images[layer + 1][i][j - 1] and
                                  cur_pixel > images[layer + 1][i][j + 1] and
                                  cur_pixel > images[layer + 1][i - 1][j] and
                                  cur_pixel > images[layer + 1][i + 1][j] and
                                  cur_pixel > images[layer + 1][i - 1][j - 1] and
                                  cur_pixel > images[layer + 1][i + 1][j + 1] and
                                  cur_pixel > images[layer + 1][i - 1][j + 1] and
                                  cur_pixel > images[layer + 1][i + 1][j - 1])
                                 or
                                 (cur_pixel < 0 and
                                  cur_pixel < images[layer][i][j - 1] and
                                  cur_pixel < images[layer][i][j + 1] and
                                  cur_pixel < images[layer][i - 1][j] and
                                  cur_pixel < images[layer][i + 1][j] and
                                  cur_pixel < images[layer][i - 1][j - 1] and
                                  cur_pixel < images[layer][i + 1][j + 1] and
                                  cur_pixel < images[layer][i - 1][j + 1] and
                                  cur_pixel < images[layer][i + 1][j - 1] and
                                  cur_pixel < images[layer - 1][i][j] and
                                  cur_pixel < images[layer - 1][i][j - 1] and
                                  cur_pixel < images[layer - 1][i][j + 1] and
                                  cur_pixel < images[layer - 1][i - 1][j] and
                                  cur_pixel < images[layer - 1][i + 1][j] and
                                  cur_pixel < images[layer - 1][i - 1][j - 1] and
                                  cur_pixel < images[layer - 1][i + 1][j + 1] and
                                  cur_pixel < images[layer - 1][i - 1][j + 1] and
                                  cur_pixel < images[layer - 1][i + 1][j - 1] and
                                  cur_pixel < images[layer + 1][i][j] and
                                  cur_pixel < images[layer + 1][i][j - 1] and
                                  cur_pixel < images[layer + 1][i][j + 1] and
                                  cur_pixel < images[layer + 1][i - 1][j] and
                                  cur_pixel < images[layer + 1][i + 1][j] and
                                  cur_pixel < images[layer + 1][i - 1][j - 1] and
                                  cur_pixel < images[layer + 1][i + 1][j + 1] and
                                  cur_pixel < images[layer + 1][i - 1][j + 1] and
                                  cur_pixel < images[layer + 1][i + 1][j - 1]
                                 )
                                )
                        ):

                            key_point = self.adjust_adjust_local_extrema(octave, layer, i, j)
                            if key_point is None:
                                continue
                            # print(key_point.x, key_point.y)
                            scl_octv = key_point.size / (1 << octave)
                            omax, hist = self.calc_orientation_hist(octave, layer, i, j, SIFT_ORI_RADIUS * scl_octv, SIFT_ORI_SIG_FCTR * scl_octv, n)

                            mag_thr = omax * 0.8
                            for k in range(n):
                                l = k - 1 if k > 0 else n - 1
                                r = k + 1 if k < n - 1 else 0
                                if hist[l] < hist[k] < hist[r] and hist[k] > mag_thr:
                                    bin = k + 0.5 * (hist[l] - hist[r]) / (hist[l] - 2 * hist[k] + hist[r])
                                    bin = n + bin if bin < 0 else (bin - n if bin >= n else bin)
                                    key_point.angle = 360 - ((360 / n) * bin)
                                    if np.abs(key_point.angle - 360) < FLT_EPSILON:
                                        key_point.angle = 0
                                    self.key_points.append(key_point)

    def adjust_adjust_local_extrema(self, octave, layer, i, j):

        img_scale = 1.0 / (255 * SIFT_FIXPT_SCALE)
        deriv_scale = img_scale * 0.5
        second_deriv_scale = img_scale
        cross_deriv_scale = img_scale * 0.25

        h, w = self.dog_pyramid[octave][layer].shape
        xc, xr, xi, contr = 0, 0, 0, 0
        for epoch in range(SIFT_MAX_INTERP_STEPS):

            up_layer = self.dog_pyramid[octave][layer + 1].astype(dtype=float)
            cur_layer = self.dog_pyramid[octave][layer].astype(dtype=float)
            down_layer = self.dog_pyramid[octave][layer - 1].astype(dtype=float)

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

            if np.abs(xi) > INT_MAX / 3 or np.abs(xr) > INT_MAX / 3 or np.abs(xc) > INT_MAX / 3:
                return None

            j += np.round(xc)
            i += np.round(xr)
            layer += np.round(xi)

            i = int(i)
            j = int(j)
            layer = int(layer)

            if layer < 1 or layer > self.n_octave_layers or i < SIFT_IMG_BORDER or i >= h - SIFT_IMG_BORDER or j < SIFT_IMG_BORDER or j >= w - SIFT_IMG_BORDER:
                return None

            if epoch == SIFT_MAX_INTERP_STEPS - 1:
                return None

            up_layer = self.dog_pyramid[octave][layer + 1]
            cur_layer = self.dog_pyramid[octave][layer]
            down_layer = self.dog_pyramid[octave][layer - 1]

            dD = np.array([
                (cur_layer[i][j + 1] - cur_layer[i][j - 1]) * deriv_scale,
                (cur_layer[i + 1][j] - cur_layer[i - 1][j]) * deriv_scale,
                (up_layer[i][j] - down_layer[i][j]) * deriv_scale
            ])

            t = np.sum(dD * np.array([xc, xr, layer]))
            contr = cur_layer[i][j] * img_scale + t * 0.5

            if np.abs(contr) * self.n_octave_layers < self.contrast_threshold:
                return None

            v2 = cur_layer[i][j] * 2
            dxx = (cur_layer[i][j + 1] - cur_layer[i][j - 1] - v2) * second_deriv_scale
            dyy = (cur_layer[i + 1][j] - cur_layer[i - 1][j] - v2) * second_deriv_scale
            dxy = (cur_layer[i - 1][j - 1] + cur_layer[i + 1][j + 1] - cur_layer[i - 1][j + 1] - cur_layer[i + 1][
                j - 1]) * cross_deriv_scale

            tr = dxx + dyy
            det = dxx * dxx - dxy * dxy
            if det <= 0 or tr * tr * self.edge_threshold >= (self.edge_threshold + 1) * (self.edge_threshold + 1) * det:
                return None

        key_point = KeyPoint()
        key_point.x = (j + xc) * 2 ** octave
        key_point.y = (i + xr) * 2 ** octave
        key_point.octave = octave
        key_point.layer = layer
        key_point.size = self.sigma * np.power(2, (layer + xi) / self.n_octave_layers) * (1 << octave)
        key_point.response = np.abs(contr)
        key_point.scale = 1 / (1 << octave) if octave >= 0 else (1 << -octave)
        return key_point

    def calc_orientation_hist(self, octave, layer, x, y, radius, sigma, n):

        img = self.dog_pyramid[octave][layer]
        rows, cols = img.shape
        length = int((radius * 2 + 1) * (radius * 2 + 1))
        expf_scale = -1.0 / (2 * sigma * sigma)
        X = []
        Y = []
        W = []
        temp_hist = [0] * (length + 2)
        # print(length)
        for i in range(-int(radius), int(radius) + 1):
            if (y+i) <= 0 or (y+i) >= rows - 1:
                continue
            for j in range(-int(radius), int(radius) + 1):
                if (x+j) <= 0 or (x+j) >= cols - 1:
                    continue
                dx = img[y+i][x+j + 1] - img[y+i][x+j - 1]
                dy = img[y+i - 1][x+j] - img[y+i + 1][x+j]
                X.append(dx)
                Y.append(dy)
                W.append((i * i + j * j) * expf_scale)
        length = len(X)
        X = np.array(X)
        Y = np.array(Y)
        W = np.array(W)
        # print(length)
        W = np.exp(W)
        Mag = np.sqrt(Y ** 2 + X ** 2)
        Ori = np.arctan2(Y, X)*180/np.pi

        for i in range(length):
            bin = int(np.round((n / 360) * Ori[i]))
            if bin >= n:
                bin -= n
            if bin < 0:
                bin += n
            temp_hist[bin] += W[i] * Mag[i]

        temp = [temp_hist[n - 1], temp_hist[n - 2], temp_hist[0], temp_hist[1]]
        temp_hist.insert(0, temp[0])
        temp_hist.insert(0, temp[1])
        temp_hist.insert(len(temp_hist), temp[2])
        temp_hist.insert(len(temp_hist), temp[3])  # padding
        hist = np.zeros(n)
        for i in range(n):
            hist[i] = (temp_hist[i - 2] + temp_hist[i + 2]) * (1 / 16) + (temp_hist[i - 1] + temp_hist[i + 1]) * (
                    4 / 16) + temp_hist[i] * (6 / 16)

        hist = np.array(hist)
        maxval = np.max(hist)
        return maxval, hist

    def calc_descriptors(self):
        d, n = SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS
        for key_point in self.key_points:
            octave, layer, scale = key_point.octave, key_point.layer, key_point.scale
            size = key_point.size * scale
            key_point.x *= scale
            key_point.y *= scale
            img = self.dog_pyramid[octave][layer]
            angle = key_point.angle
            if np.abs(angle - 360) < FLT_EPSILON:
                angle = 0
            self.calc_sift_descriptor(img, key_point, angle, size * 0.5, d, n)

    def calc_sift_descriptor(self, img, key_point, ori, scl, d, n):
        h, w = img.shape
        x, y = int(np.round(key_point.x)), int(np.round(key_point.y))
        cos_t = np.cos(ori * (np.pi / 180))
        sin_t = np.sin(ori * (np.pi / 180))

        bins_per_rad = n / 360
        exp_scale = -1.0 / (d * d * 0.5)
        hist_width = SIFT_DESCR_SCL_FCTR * scl

        radius = int(np.round(hist_width * np.sqrt(2) * (d + 1) * 0.5))
        radius = min(radius, int(np.sqrt(h * h + w * w)))

        cos_t /= hist_width
        sin_t /= hist_width

        hist_length = (d + 2) * (d + 2) * (n + 2)
        hist = [0] * hist_length

        X, Y, RBin, CBin, W = [], [], [], [], []

        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                c_rot = j * cos_t - i * sin_t
                r_rot = j * sin_t + i * cos_t

                rbin = r_rot + d / 2 - 0.5
                cbin = c_rot + d / 2 - 0.5

                r = y + i
                c = x + j

                if -1 < rbin < d and -1 < cbin < d and 0 < r < h - 1 and 0 < c < w - 1:
                    dx = img[r][c + 1] - img[r][c - 1]
                    dy = img[r - 1][c] - img[r + 1][c]

                    X.append(dx)
                    Y.append(dy)
                    RBin.append(rbin)
                    CBin.append(cbin)
                    W.append((c_rot ** 2 + r_rot ** 2) * exp_scale)

        length = len(X)
        X = np.array(X).reshape(length, 1)
        Y = np.array(Y).reshape(1, length)
        W = np.exp(W)
        Ori = np.arctan2(Y, X)*180/np.pi
        Mag = np.sqrt(Y ** 2 + X ** 2)

        for i in range(length):
            rbin, cbin = RBin[i], CBin[i]
            obin = (Ori[i] - ori) * bins_per_rad
            print(obin)
            mag = Mag[i] * W[i]

            r0 = int(rbin)
            c0 = int(cbin)
            o0 = int(obin)

            rbin -= r0
            cbin -= c0
            obin -= o0

            if o0 < 0:
                o0 += n
            if o0 >= n:
                o0 -= n

            v_r1 = mag * rbin
            v_r0 = mag - v_r1
            v_rc11 = v_r1 * cbin
            v_rc10 = v_r1 - v_rc11
            v_rc01 = v_r0 * cbin
            v_rc00 = v_r0 - v_rc01
            v_rco111 = v_rc11 * obin
            v_rco110 = v_rc11 - v_rco111
            v_rco101 = v_rc10 * obin
            v_rco100 = v_rc10 - v_rco101
            v_rco011 = v_rc01 * obin
            v_rco010 = v_rc01 - v_rco011
            v_rco001 = v_rc00 * obin
            v_rco000 = v_rc00 - v_rco001

            idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0

            hist[idx] += v_rco000
            hist[idx + 1] += v_rco001
            hist[idx + (n + 2)] += v_rco010
            hist[idx + (n + 3)] += v_rco011
            hist[idx + (d + 2) * (n + 2)] += v_rco100
            hist[idx + (d + 2) * (n + 2) + 1] += v_rco101
            hist[idx + (d + 3) * (n + 2)] += v_rco110
            hist[idx + (d + 3) * (n + 2) + 1] += v_rco111

        dst = np.zeros(d * d * n)
        for i in range(d):
            for j in range(d):
                idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2)
                hist[idx] += hist[idx + n]
                hist[idx + 1] += hist[idx + n + 1]
                for k in range(n):
                    dst[(i * d + j) * n + k] = hist[idx + k]

        nrm2 = 0
        length = d * d * n
        for i in range(length):
            nrm2 += dst[i] * dst[i]
        thr = np.sqrt(nrm2) * SIFT_DESCR_MAG_THR
        nrm2 = 0
        for i in range(len(length)):
            val = min(dst[i], thr)
            dst[i] = val
            nrm2 += val * val

        nrm2 = SIFT_INT_DESCR_FCTR / max(np.sqrt(nrm2), FLT_EPSILON)

        for i in range(length):
            dst[i] = dst[i] * nrm2

        self.descriptors.append(dst)
