import numpy as np


def calculate_psr(response_map, exclude_radius, rolled=False):
    h, w = response_map.shape

    if rolled:
        response_map = np.roll(response_map, -int(np.floor(w / 2)), axis=1)
        response_map = np.roll(response_map, -int(np.floor(h / 2)), axis=0)

    y_peak, x_peak = np.unravel_index(np.argmax(response_map), (h, w))
    mask = np.ones_like(response_map, dtype=bool)
    mask[
        max(0, y_peak - exclude_radius) : min(y_peak + exclude_radius + 1, h),
        max(0, x_peak - exclude_radius) : min(x_peak + exclude_radius + 1, w),
    ] = False
    sidelobe = response_map[mask]

    psr = (response_map[y_peak, x_peak] - np.mean(sidelobe)) / (np.std(sidelobe) + 1e-6)
    return psr
