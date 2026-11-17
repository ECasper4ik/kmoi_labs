"""Лаба 2. Эквализация: LUT[i] = 255 * SUM(Hist, Imin, i) / SUM(Hist, Imin, Imax) (CDF)."""
import cv2
import numpy as np

import common
from lab01_normalization import normalize


def equalize(gray, opencv_style=False):
    """Эквализация через LUT. opencv_style: самый тёмный пиксель -> ровно 0."""
    common.check(gray)
    cdf = np.cumsum(np.bincount(gray.ravel(), minlength=256)).astype(float)
    lo = cdf[gray.min()] if opencv_style else 0.0  # CDF в Imin = число пикселей на Imin
    if gray.min() == gray.max():
        return gray.copy()
    lut = 255.0 * (cdf - lo) / (cdf[-1] - lo)
    return np.clip(np.rint(lut), 0, 255).astype(np.uint8)[gray]


def main():
    args = common.parse(__doc__)
    gray = common.load(args.image) if args.image else (common.demo_objects()[0] * 0.35 + 60).astype(np.uint8)
    eq = equalize(gray)
    ref = cv2.equalizeHist(gray)
    print("max|diff| (вариант OpenCV) с cv2.equalizeHist:", np.abs(equalize(gray, True).astype(int) - ref).max())
    print(f"std: исходное {gray.std():.1f}, нормализация {normalize(gray).std():.1f}, эквализация {eq.std():.1f}")
    print(f"время: наш {common.ms(equalize, gray):.3f} мс, cv2 {common.ms(cv2.equalizeHist, gray):.3f} мс")
    noisy = common.gauss(gray, 6)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(noisy)
    print(f"усиление шума: std(eq(шум) - eq(чистое)) = {(equalize(noisy).astype(float) - eq).std():.1f}")
    common.grid([(gray, "исходное"), (eq, "эквализация"), (equalize(noisy), "эквализация + шум"),
                 (clahe, "CLAHE + шум")], args.out, "lab02.png", 4)
    cdf = lambda g: np.cumsum(np.bincount(g.ravel(), minlength=256)) / g.size  # noqa: E731
    common.plot({"CDF исходного": cdf(gray), "CDF после эквализации": cdf(eq)}, args.out, "lab02_cdf.png")


if __name__ == "__main__":
    main()
