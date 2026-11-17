"""Лаба 4. Проекции: PH(y) = sum_x I(x, y), PV(x) = sum_y I(x, y)."""
import cv2
import numpy as np

import common
from lab05_binarization import otsu


def projections(binary):
    """(PH, PV) бинарного изображения (объект = ненулевой); int64 против переполнения."""
    common.check(binary)
    obj = (binary > 0).astype(np.int64)
    return obj.sum(axis=1), obj.sum(axis=0)


def runs(p, thr=0, min_len=1):
    """Участки (начало, конец), где p > thr -- например, строки текста."""
    d = np.flatnonzero(np.diff(np.r_[0, (p > thr).astype(int), 0]))
    return [(int(s), int(e)) for s, e in zip(d[::2], d[1::2]) if e - s >= min_len]


def rotate(img, angle):
    m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1.0)
    return cv2.warpAffine(img, m, img.shape[::-1])


def skew(binary, max_angle=10, step=0.5):
    """Угол поворота, при котором дисперсия PH максимальна (строки выровнены)."""
    angles = np.arange(-max_angle, max_angle + step, step)
    return float(angles[np.argmax([np.var(projections(rotate(binary, a))[0]) for a in angles])])


def main():
    args = common.parse(__doc__)
    gray = common.load(args.image) if args.image else common.demo_text(3.0)
    binary = (gray <= otsu(gray)).astype(np.uint8) * 255  # объект (тёмный текст) = белый
    ph, pv = projections(binary)
    ref = cv2.reduce((binary > 0).astype(np.float64), 1, cv2.REDUCE_SUM).ravel()
    print("max|PH - cv2.reduce|:", np.abs(ph - ref).max())
    angle = skew(binary)
    fixed = rotate(binary, angle)
    print(f"угол коррекции {angle:+.1f}°, строки текста: {runs(projections(fixed)[0], 0.02 * ph.max(), 5)}")
    print(f"время: projections {common.ms(projections, binary):.3f} мс, skew {common.ms(skew, binary, repeat=1):.0f} мс")
    noisy = common.salt_pepper(binary, 0.02)
    clean = cv2.medianBlur(noisy, 3)
    print(f"шум 2%: RMS(PH) {np.sqrt(np.mean((projections(noisy)[0] - ph) ** 2)):.1f}, "
          f"после медианы {np.sqrt(np.mean((projections(clean)[0] - ph) ** 2)):.1f}")
    common.grid([(binary, "бинарное"), (fixed, "после компенсации угла")], args.out, "lab04.png", 2)
    common.plot({"PH до": ph, "PH после": projections(fixed)[0]}, args.out, "lab04_ph.png")


if __name__ == "__main__":
    main()
