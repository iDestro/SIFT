import numpy as np
import cv2


class SIFT:
    def __int__(self, n_octaves, n_octave_layers, contrast_threshold, edge_threshold, sigma):
        self.n_octaves = n_octaves
        self.n_octave_layers = n_octave_layers
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        self.key_points = []
        self.descriptors = []
        self.gaussian_pyramid = {i: [] for i in range(self.n_octaves)}
        self.dog_pyramid = {i: [] for i in range(self.n_octaves)}

    def build_gaussian_pyramid(self, n_octaves, img):
        for octave in range(n_octaves):
            if octave == 0:
                cur_octave_init_img = img
            else:
                cur_octave_init_img = self.gaussian_pyramid[octave - 1][2]
                w, h = cur_octave_init_img.shape
                cur_octave_init_img = cv2.resize(cur_octave_init_img, (h // 2, w // 2))
            for i in range(self.n_octave_layers + 3):
                cur_sigma = np.power(self.sigma, octave + i / (self.n_octave_layers + 3))
                size = (int(6 * cur_sigma + 1), int(6 * cur_sigma + 1))
                temp_img = cv2.GaussianBlur(cur_octave_init_img, ksize=size, sigmaX=cur_sigma, sigmaY=cur_sigma)
                self.gaussian_pyramid[octave].append(temp_img)

    def build_dog_pyramid(self):
        for octave, layer in self.gaussian_pyramid.items():
            for i in range(0, self.n_octave_layers + 2):
                self.dog_pyramid[octave].append(layer[i + 1] - layer[i])

    def find_scale_space_extrema(self):
        threshold = int(np.round(0.5 * self.contrast_threshold / self.n_octave_layers * 255))
        n = 36
        hist = None
        for octave, images in self.dog_pyramid.items():
            for layer in range(1, len(images) - 1):
                h, w = images[layer].shape
                for i in range(1, h - 1):
                    for j in range(1, w - 1):
                        cur_pixel = images[layer][i][j]
                        if (
                                cur_pixel > 0 and
                                cur_pixel > images[layer][i][j - 1] and cur_pixel > images[layer][i][j + 1] and
                                cur_pixel > images[layer][i - 1][j] and cur_pixel > images[layer][i + 1][j] and
                                cur_pixel > images[layer][i - 1][j - 1] and cur_pixel > images[layer][i + 1][j + 1] and
                                cur_pixel > images[layer][i - 1][j + 1] and cur_pixel > images[layer][i + 1][j - 1] and

                                cur_pixel > images[layer - 1][i][j] and
                                cur_pixel > images[layer - 1][i][j - 1] and cur_pixel > images[layer - 1][i][j + 1] and
                                cur_pixel > images[layer - 1][i - 1][j] and cur_pixel > images[layer - 1][i + 1][j] and
                                cur_pixel > images[layer - 1][i - 1][j - 1] and cur_pixel > images[layer - 1][i + 1][
                                    j + 1] and
                                cur_pixel > images[layer - 1][i - 1][j + 1] and cur_pixel > images[layer - 1][i + 1][
                                    j - 1] and

                                cur_pixel > images[layer + 1][i][j] and
                                cur_pixel > images[layer + 1][i][j - 1] and cur_pixel > images[layer + 1][i][j + 1] and
                                cur_pixel > images[layer + 1][i - 1][j] and cur_pixel > images[layer + 1][i + 1][j] and
                                cur_pixel > images[layer + 1][i - 1][j - 1] and cur_pixel > images[layer + 1][i + 1][
                                    j + 1] and
                                cur_pixel > images[layer + 1][i - 1][j + 1] and cur_pixel > images[layer + 1][i + 1][
                                    j - 1]
                        ) or (
                                cur_pixel < 0 and
                                cur_pixel < images[layer][i][j - 1] and cur_pixel < images[layer][i][j + 1] and
                                cur_pixel < images[layer][i - 1][j] and cur_pixel < images[layer][i + 1][j] and
                                cur_pixel < images[layer][i - 1][j - 1] and cur_pixel < images[layer][i + 1][j + 1] and
                                cur_pixel < images[layer][i - 1][j + 1] and cur_pixel < images[layer][i + 1][j - 1] and

                                cur_pixel < images[layer - 1][i][j] and
                                cur_pixel < images[layer - 1][i][j - 1] and cur_pixel < images[layer - 1][i][j + 1] and
                                cur_pixel < images[layer - 1][i - 1][j] and cur_pixel < images[layer - 1][i + 1][j] and
                                cur_pixel < images[layer - 1][i - 1][j - 1] and cur_pixel < images[layer - 1][i + 1][
                                    j + 1] and
                                cur_pixel < images[layer - 1][i - 1][j + 1] and cur_pixel < images[layer - 1][i + 1][
                                    j - 1] and

                                cur_pixel < images[layer + 1][i][j] and
                                cur_pixel < images[layer + 1][i][j - 1] and cur_pixel < images[layer + 1][i][j + 1] and
                                cur_pixel < images[layer + 1][i - 1][j] and cur_pixel < images[layer + 1][i + 1][j] and
                                cur_pixel < images[layer + 1][i - 1][j - 1] and cur_pixel < images[layer + 1][i + 1][
                                    j + 1] and
                                cur_pixel < images[layer + 1][i - 1][j + 1] and cur_pixel < images[layer + 1][i + 1][
                                    j - 1]
                        ):

                            key_point = self.adjust_adjust_local_extrema(octave, layer, i, j)
                            if key_point is None:
                                continue
                            scl_octv = key_point[3] * 0.5 / (1 << octave)
                            omax, hist = self.calc_orientation_hist(octave, layer, i, j, 3 * scl_octv, 1.5 * scl_octv, n)
                            mag_thr = omax*0.8
                            for i in range(n):
                                l = j-1 if i > 0 else n-1
                                r = j+1 if i < n-1 else 0
                                if hist[l] < hist[i] < hist[r] and hist[i] > mag_thr:
                                    bin = i + 0.5*(hist[l]-hist[r])/(hist[l]-2*hist[j]+hist[r])
                                    bin = n+bin if bin < 0 else (bin-n if bin >= n else bin)
                                    key_point.append(360 - ((360/n) * bin))
                                    if np.abs(key_point[-1]-360 < 0.0001):
                                        key_point[-1] = 0
                                    self.key_points.append(key_point)

    def adjust_adjust_local_extrema(self, octave, layer, i, j):

        img_scale = 1.0 / 255
        deriv_scale = img_scale * 0.5
        second_deriv_scale = img_scale
        cross_deriv_scale = img_scale * 0.25

        INT_MAX = 2147483647
        SIFT_MAX_INTERP_STEPS = 5
        SIFT_IMG_BORDER = 5

        up_layer = self.dog_pyramid[octave][layer + 1].astype(dtype=float)
        cur_layer = self.dog_pyramid[octave][layer].astype(dtype=float)
        down_layer = self.dog_pyramid[octave][layer - 1].astype(dtype=float)

        h, w = cur_layer.shape
        xc, xr, xi, contr = 0, 0, 0, 0
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
                return None

            i += np.round(xc)
            j += np.round(xr)
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

        x = (i + xc) * 2 ** octave
        y = (j + xr) * 2 ** octave
        o = octave + (layer << 8) + (np.round((xi + 0.5) * 255) << 16)
        size = self.sigma * np.power(2, (layer + xi) / self.n_octave_layers) * (1 << octave) * 2
        response = np.abs(contr)
        return [x, y, o, size, response]

    def calc_orientation_hist(self, octave, layer, x, y, radius, sigma, n):

        img = self.dog_pyramid[octave][layer]
        cols, rows = img.shape
        length = (radius * 2 + 1) * (radius * 2 + 1)
        expf_scale = -1.0 / (2 * sigma * sigma)
        X = [0] * length
        Y = [0] * length
        W = [0] * length
        temp_hist = [0] * (length + 2)

        k = 0
        for i in range(-radius, radius + 1):
            if y <= 0 or y >= rows - 1:
                continue
            for j in range(-radius, radius + 1):
                if x <= 0 or x >= cols - 1:
                    continue
                dx = img[y][x + 1] - img[y][x - 1]
                dy = img[y - 1][x] - img[y + 1][x]

                X[k] = dx
                Y[k] = dy
                W[k] = (i * i + j * j) * expf_scale
                k += 1
        length = k
        X = np.array(X[:length]).reshape(1, length)
        Y = np.array(Y[:length]).reshape(length, 1)
        W = np.array(W[:length])
        Ori = np.arctan2(Y, X)
        Mag = np.sqrt(Y ** 2 + X ** 2)

        for i in range(length):
            bin = int(np.round((n / 360) * Ori[i]))
            if bin >= n:
                bin -= n
            if bin < 0:
                bin += n
            temp_hist[bin] += W[i] * Mag[i, :]

        temp_hist[-1] = temp_hist[n - 1]
        temp_hist[-2] = temp_hist[n - 2]
        temp_hist[n] = temp_hist[0]
        temp_hist[n + 1] = temp_hist[1]
        hist = np.zeros(length)
        for i in range(n):
            hist[i] = (temp_hist[i - 2] + temp_hist[i + 2]) * (1 / 16) + (temp_hist[i - 1] + temp_hist[i + 1]) * (
                        4 / 16) + temp_hist[i] * (6 / 16)

        hist = np.array(hist)
        maxval = np.max(hist)
        return maxval, hist

    def calc_descriptors(self):
        d, n = 4, 8
        octave, layer , scale = 0, 0, 0
        for key_point in self.key_points:
            size = key_point[3]*scale
            key_point[0] *= scale
            key_point[1] *= scale
            img = self.dog_pyramid[octave][layer]
            angle = key_point[-1]
            if np.abs(angle-360) < 0.0001:
                angle = 0
            self.calc_sift_descriptor(img, key_point, angle, size*0.5, d, n)

    def calc_sift_descriptor(self, img, key_point, ori, scl, d, n):
        h, w = img.shape
        x, y = int(key_point[0]), int(key_point[1])
        cos_t = np.cos(ori*(np.pi/180))
        sin_t = np.sin(ori*(np.pi/180))

        bins_per_rad = n / 360
        exp_scale = -1.0 / (d*d*0.5)
        hist_width = 3*scl

        radius = int(np.round(hist_width*np.sqrt(2)*(d+1)*0.5))
        radius = min(radius, int(np.sqrt(h*h+w*w)))

        cos_t /= hist_width
        sin_t /= hist_width

        length = (radius*2+1)*(radius*2+1)
        hist_length = (d+2)*(d+2)*(n+2)
        hist = [0]*hist_length

        X, Y, RBin, CBin, W = [], [], [], [], []
        k = 0
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                c_rot = j * cos_t - i * sin_t
                r_rot = j * sin_t + i * cos_t

                rbin = r_rot + d/2 - 0.5
                cbin = c_rot + d/2 - 0.5

                r = y+i
                c = x+j

                if -1 < rbin < d and -1 < cbin < d and 0 < r < h-1 and 0 < c < w-1:

                    dx = img[r][c+1]-img[r][c-1]
                    dy = img[r-1][c]-img[r+1][c]

                    X.append(dx)
                    Y.append(dy)
                    RBin.append(rbin)
                    CBin.append(cbin)
                    W.append((c_rot**2+r_rot**2)*exp_scale)
                    k += 1

        length = k
        X = np.array(X).reshape(length, 1)
        Y = np.array(Y).reshape(1, length)
        Ori = np.arctan2(Y, X)
        Mag = Ori = np.arctan2(Y, X)

        for i in range(length):
            rbin, cbin = RBin[i], CBin[i]
            obin = (Ori[k] - ori)*bins_per_rad
            mag = Mag[k]*W[k]

            r0 = np.round(rbin)
            c0 = np.round(cbin)
            o0 = np.round(obin)

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

        for i in range(d):
            for j in range(d):
                idx = ((i+1)*(d+2)+(j+1))*(n+2)
                hist[idx] += hist[idx+n]
                hist[idx+1] += hist[idx+n+1]
                for k in range(n):
                    dst[(i*d+j)*n+k] = hist[idx+k]

        nrm2 = 0
        length = d*d*n
        for i in range(length):
            nrm2 += dst[k]*dst[k]
        thr = np.sqrt(nrm2)*SIFT_DESCR_MAG_THR

        for i in range(len(length)):
            val = min(dst[i], thr)
            dst[i] = val
            nrm2 += val*val

        nrm2 = SIFT_INT_DESCR_FCTR/max(np.sqrt(nrm2), FLT_EPSILON)

        for i in range(length):
            dst[i] = dst[i]*nrm2

