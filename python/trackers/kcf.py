import numpy as np
from dataclasses import dataclass
import cv2
from numpy.fft import fft2, ifft2

from confidence.psr import calculate_psr
from trackers.base_tracker import BaseTracker

@dataclass
class KCFParams:
    features: str = "gray"
    lambda_reg: float = 1e-4
    gamma: float = 0.075
    padding: float = 1.5
    kernel: str = "rbf"
    rbf_sigma: float = 0.1
    debug: bool = True


def ufft2(image):
    return fft2(image, axes=(0, 1))

def uifft2(image):
    return ifft2(image, axes=(0,1))

def create_hanning_window(w, h):
    hann1d_w = np.hanning(w)[None, :]
    hann1d_h = np.hanning(h)[:, None]
    hann2d = hann1d_h.dot(hann1d_w)
    return hann2d[:, :, None]


class KCFTracker(BaseTracker):
    def __init__(self, params: KCFParams):
        self.params = params
        self.lambda_reg = params.lambda_reg
        self.sigma = params.rbf_sigma
        self.alpha = None
        self.x = None
        self.window = None
        self.xc, self.yc = None, None # center of a target
        self.target_width, self.target_height = None, None
        self.gamma = params.gamma
        self.debug = params.debug
        self._scales = [0.95, 1.0, 1.05]
        self._current_scale_factor = 1.0
        self._new_scale = 1.0
    
    def init(self, image, bbox):
        assert bbox[2] >= 0 and bbox[3] >= 0
        self.pos = list(bbox)      
        x, y, w, h = map(int, bbox)
        self.target_width = w 
        self.target_height = h

        self.padded_width = int(np.floor(self.target_width * (1 + self.params.padding)))
        self.padded_height = int(np.floor(self.target_height * (1 + self.params.padding)))
        self.base_padded_width = int(np.floor(self.target_width * (1 + self.params.padding)))
        self.base_padded_height = int(np.floor(self.target_height * (1 + self.params.padding)))

        self.xc = np.floor(x + w / 2)
        self.yc = np.floor(y + h / 2)

        self.x = self.extract_patch(image)
        self.x = self.extract_features(self.x)

        self.window = create_hanning_window(self.base_padded_width, self.base_padded_height)
        self.y_label_fft = ufft2(self.calculate_label(self.base_padded_width, self.base_padded_height))
        self.x = self.x * self.window
        self.alpha = self._train(self.x)

        if self.debug:
            cv2.imshow("x", self.x)

    def extract_features(self, image):
        if self.params.features == "gray":
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_gray = image_gray.astype(float)[:, :, None] / 255.0
            image_gray = image_gray - np.mean(image_gray)
            return image_gray
        elif self.params.features == "rgb":
            assert image.ndim == 3 and image.shape[2] == 3
            image_rgb = image.astype(float) / 255.0
            image_rgb = image_rgb - np.mean(image_rgb, axis=(0, 1), keepdims=True)
            return image_rgb
        
        raise Exception(f"Unsupported feature type {self.params.features}")
        
    def extract_patch(self, image):
        center = int(self.xc), int(self.yc)
        cropped = cv2.getRectSubPix(image, (int(self.padded_width), int(self.padded_height)), center)
        interpolation = cv2.INTER_AREA if self._current_scale_factor > 1.0 else cv2.INTER_LINEAR
        cropped = cv2.resize(cropped, (self.base_padded_width, self.base_padded_height), interpolation=interpolation)
        return cropped

    def update(self, image):
        z = self.extract_patch(image)
        z = self.extract_features(z)
        z = z * self.window

        if self.debug:
            cv2.imshow("z", z)

        k = self._kernel_correlation(z, self.x)
        responses = np.real(uifft2(self.alpha * ufft2(k)))

        yc, xc = np.unravel_index(np.argmax(responses), responses.shape)

        if self.debug:
            cv2.imshow("response", responses)

        if yc + 1> self.base_padded_height / 2:
             yc -= self.base_padded_height
        if xc + 1> self.base_padded_width / 2:
             xc -= self.base_padded_width

        self.xc = np.floor(self.xc + xc * self._current_scale_factor)
        self.yc = np.floor(self.yc + yc * self._current_scale_factor)

        self.pos[0] = self.xc - self.target_width//2 
        self.pos[1] = self.yc - self.target_height//2

        # scale estimation
        result_response_map = self._update_scale(image)
        confidence = calculate_psr(result_response_map, exclude_radius=4, rolled=True)

        self.target_width = np.floor(self.target_width * self._new_scale)
        self.target_height = np.floor(self.target_height * self._new_scale)
        self.padded_width = np.floor(self.padded_width * self._new_scale)
        self.padded_height = np.floor(self.padded_height * self._new_scale)

        self.pos[2] = self.target_width 
        self.pos[3] = self.target_height

        x_new = self.extract_patch(image)
        x_new = self.extract_features(x_new)
        x_new = x_new * self.window
        alpha_new = self._train(x_new)

        self.x = self.gamma * x_new + (1 - self.gamma) * self.x
        self.alpha = self.gamma * alpha_new + (1 - self.gamma) * self.alpha

        if self.debug:
            cv2.imshow("x", self.x)

        return confidence, self.pos

    def _train(self, x):
        k = self._kernel_correlation(x, x)
        alpha = self.y_label_fft / (ufft2(k) + self.lambda_reg)
        return alpha
    
    def calculate_label(self, label_width, label_height):
        output_sigma = np.sqrt(self.target_height * self.target_width) * 0.1
        grid_x, grid_y = np.meshgrid(np.arange(label_width) - label_width//2, np.arange(label_height) - label_height//2)
        labels = np.exp(-0.5 * (grid_x ** 2 + grid_y ** 2) / (output_sigma ** 2))
        labels = np.roll(labels, -int(np.floor(label_width / 2)), axis=1)
        labels = np.roll(labels,-int(np.floor(label_height / 2)),axis=0)
        assert labels[0, 0] == 1

        if self.debug:
            cv2.imshow("y", labels)

        return labels

    def _kernel_correlation(self, x1, x2):
        x1_f, x2_f = ufft2(x1), ufft2(x2)
        N = x1_f.shape[0] * x1_f.shape[1]
        c = uifft2(x1_f * x2_f.conj()) # sum over feature channels in the future
        c = np.real(c).sum(axis=2)
        d = np.sum(x1_f.conj() * x1_f) / N + np.sum(x2_f.conj() * x2_f) / N - 2.0 * c
        k = np.exp(-1.0 / self.sigma**2 * np.clip(d, a_min=0, a_max=None) / d.size)
        return k
    
    def create_scaled_images(self, image):
        scaled_images = []
        for scale in self._scales:
            scaled_roi = cv2.getRectSubPix(image, (int(self.padded_width * scale), int(self.padded_height * scale)), (self.xc, self.yc))
            interpolation = cv2.INTER_AREA if scale > 1.0 else cv2.INTER_LINEAR
            scaled_roi = cv2.resize(scaled_roi, (self.base_padded_width, self.base_padded_height), interpolation=interpolation)
            scaled_images.append(scaled_roi)
        return scaled_images
    
    def _update_scale(self, image):
        scaled_examples = self.create_scaled_images(image)
        max_corr = -np.inf
        result_response_map = None
        for z, scale in zip(scaled_examples, self._scales):
            z = self.extract_features(z)
            z = z * self.window
            k = self._kernel_correlation(z, self.x)
            responses = np.real(uifft2(self.alpha * ufft2(k)))
            if responses.max() > max_corr:
                max_corr = responses.max()
                result_response_map = responses
                self._new_scale = scale

        self._current_scale_factor *= self._new_scale
        return result_response_map
