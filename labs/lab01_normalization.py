"""Лаба 1. Яркостная нормализация: LUT[i] = 255 * (i - Imin) / (Imax - Imin)."""
import cv2
import numpy as np

import common


def make_lut(lo, hi):
    """LUT растяжения [lo, hi] -> [0, 255]; при hi <= lo -- тождественная."""
    if hi <= lo:
        return np.arange(256, dtype=np.uint8)
    # float64: (i - lo) * 255 в uint8 переполнилось бы; rint, а не усечение
    return np.clip(np.rint(255.0 * (np.arange(256) - lo) / (hi - lo)), 0, 255).astype(np.uint8)


def normalize(gray, lo=None, hi=None):
    """Нормализация по min/max изображения (или заданным lo, hi)."""
    common.check(gray)
    lo = gray.min() if lo is None else lo
    hi = gray.max() if hi is None else hi
    return make_lut(int(lo), int(hi))[gray]


def normalize_robust(gray, low=1, high=99):
    """Границы по перцентилям: выброс не задаёт Imax (хвосты насыщаются)."""
    lo, hi = np.percentile(gray, [low, high])
    return normalize(gray, round(lo), round(hi))


def normalize_color(bgr):
    """Нормализация яркости V в HSV: цветовой баланс не меняется."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = normalize(np.ascontiguousarray(hsv[..., 2]))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main():
    args = common.parse(__doc__)
    if args.image:
        gray = common.load(args.image)
    else:
        gray = (common.demo_objects()[0] * 0.3 + 90).astype(np.uint8)
    mine = normalize(gray)
    ref = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    print("max|diff| с cv2.normalize:", np.abs(mine.astype(int) - ref).max())
    print(f"время: наш {common.ms(normalize, gray):.3f} мс, cv2 {common.ms(cv2.normalize, gray, None, 0, 255, cv2.NORM_MINMAX):.3f} мс")
    noisy = gray.copy()
    noisy[0, 0] = 255  # один выброс
    print(f"std: чистое {mine.std():.1f}, с выбросом {normalize(noisy).std():.1f}, "
          f"перцентили {normalize_robust(noisy).std():.1f}")
    common.grid([(gray, "исходное"), (mine, "нормализация"), (noisy, "+выброс 255"),
                 (normalize(noisy), "нормализация (выброс)"), (normalize_robust(noisy), "перцентили")],
                args.out, "lab01.png")


if __name__ == "__main__":
    main()
