"""Лаба 5. Бинаризация: порог, Оцу, ISODATA, локальный порог Саувола."""
import cv2
import numpy as np

import common


def threshold(gray, t):
    """255 где I > t, иначе 0."""
    common.check(gray)
    return (gray > t).astype(np.uint8) * 255


def otsu(gray):
    """Порог Оцу: максимум sigma_b^2(t) = (mu_T*w(t) - mu(t))^2 / (w(t) * (1 - w(t)))."""
    common.check(gray)
    p = np.bincount(gray.ravel(), minlength=256) / gray.size
    w, mu = np.cumsum(p), np.cumsum(p * np.arange(256))
    d = w * (1 - w)
    s = np.zeros(256)
    s[d > 1e-12] = (mu[-1] * w[d > 1e-12] - mu[d > 1e-12]) ** 2 / d[d > 1e-12]
    return int(np.argmax(s))


def isodata(gray, eps=0.5):
    """Итеративный порог: T <- (mean(I <= T) + mean(I > T)) / 2."""
    t = float(gray.mean())
    for _ in range(100):
        lo, hi = gray[gray <= t], gray[gray > t]
        if lo.size == 0 or hi.size == 0:
            break
        new = (lo.mean() + hi.mean()) / 2
        done = abs(new - t) < eps
        t = new
        if done:
            break
    return int(round(t))


def box_sum(img, win):
    """Суммы по окнам win x win через интегральное изображение (O(N) при любом win)."""
    h, w = img.shape
    ii = np.zeros((h + win, w + win))
    ii[1:, 1:] = np.pad(img.astype(float), win // 2, mode="reflect").cumsum(0).cumsum(1)
    return ii[win:win + h, win:win + w] - ii[:h, win:win + w] - ii[win:win + h, :w] + ii[:h, :w]


def sauvola(gray, win=31, k=0.2, bright_objects=True):
    """Локальный порог T = m * (1 + k * (s / 128 - 1)); окно должно быть >= 2x объекта."""
    common.check(gray)
    g = 255 - gray if bright_objects else gray  # метод рассчитан на тёмный объект
    m = box_sum(g, win) / win ** 2
    s = np.sqrt(np.maximum(box_sum(g.astype(float) ** 2, win) / win ** 2 - m ** 2, 0))
    return (g <= m * (1 + k * (s / 128 - 1))).astype(np.uint8) * 255


def main():
    args = common.parse(__doc__)
    gray, truth = (common.load(args.image), None) if args.image else common.demo_objects()
    t = otsu(gray)
    print(f"Оцу {t}, cv2 {int(cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[0])}, ISODATA {isodata(gray)}")
    b_otsu, b_loc = threshold(gray, t), sauvola(gray, 61)
    ref = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 61, -10)
    print(f"время: Оцу {common.ms(otsu, gray):.3f} мс, Саувола {common.ms(sauvola, gray, 61):.2f} мс")
    noisy = common.gauss(gray, 25)
    b_n, b_f = sauvola(noisy, 61), sauvola(cv2.medianBlur(noisy, 5), 61)
    if truth is not None:
        print(f"IoU: Оцу {common.iou(b_otsu > 0, truth):.3f}, Саувола {common.iou(b_loc > 0, truth):.3f}, "
              f"cv2.adaptive {common.iou(ref > 0, truth):.3f}; шум s=25: {common.iou(b_n > 0, truth):.3f}, "
              f"после медианы {common.iou(b_f > 0, truth):.3f}")
    common.grid([(gray, "исходное"), (b_otsu, "Оцу"), (b_loc, "Саувола"), (ref, "cv2.adaptiveThreshold"),
                 (b_n, "Саувола + шум"), (b_f, "медиана + Саувола")], args.out, "lab05.png")


if __name__ == "__main__":
    main()
