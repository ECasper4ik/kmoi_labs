"""Лаба 6. Нелинейная фильтрация: ранговые фильтры, минимакс, нормализация фона."""
import cv2
import numpy as np

import common


def rank_filter(gray, k, rank):
    """Элемент с номером rank (0..k*k-1) в упорядоченном окне k x k (края -- reflect)."""
    common.check(gray)
    if k % 2 == 0 or not 0 <= rank < k * k:
        raise ValueError("k нечётное, 0 <= rank < k*k")
    flat = common.windows(gray, k).reshape(*gray.shape, k * k)
    return np.partition(flat, rank, axis=-1)[..., rank]  # частичная сортировка, не полная


def median(gray, k):
    return rank_filter(gray, k, k * k // 2)


def extreme(gray, k, func):
    """min/max окна k x k двумя 1D-проходами (разделимо: O(N*k) вместо O(N*k^2))."""
    r = k // 2
    sw = np.lib.stride_tricks.sliding_window_view
    tmp = func(sw(np.pad(gray, ((0, 0), (r, r)), mode="reflect"), k, axis=1), axis=-1)
    return func(sw(np.pad(tmp, ((r, r), (0, 0)), mode="reflect"), k, axis=0), axis=-1)


def minimax(gray, k):
    """min, затем max: убирает светлые выбросы меньше окна."""
    return extreme(extreme(gray, k, np.min), k, np.max)


def background_normalization(gray, k_small=3, k_big=35):
    """Малая апертура -- шум, большая -- объекты (остаётся фон); объекты = изображение - фон."""
    smooth = median(gray, k_small)
    background = minimax(smooth, k_big)
    return np.clip(smooth.astype(int) - background, 0, 255).astype(np.uint8), background


def main():
    args = common.parse(__doc__)
    gray, truth = (common.load(args.image), None) if args.image else common.demo_objects(noise=0)
    k, inner = 5, (slice(5, -5),) * 2
    diff = np.abs(median(gray, k)[inner].astype(int) - cv2.medianBlur(gray, k)[inner])
    ones = np.ones((k, k), np.uint8)
    print(f"max|diff| внутри: медиана {diff.max()}, min {np.abs(extreme(gray, k, np.min)[inner].astype(int) - cv2.erode(gray, ones)[inner]).max()}")
    noisy = common.salt_pepper(gray, 0.10)
    print(f"PSNR при шуме 10%: шум {common.psnr(gray, noisy):.1f}, среднее {common.psnr(gray, cv2.blur(noisy, (k, k))):.1f}, "
          f"медиана {common.psnr(gray, median(noisy, k)):.1f} дБ")
    for kk in (3, 9, 15):
        print(f"время k={kk}: медиана {common.ms(median, gray, kk, repeat=1):.1f} мс, cv2 {common.ms(cv2.medianBlur, gray, kk):.2f} мс, "
              f"min (разделимо) {common.ms(extreme, gray, kk, np.min, repeat=1):.2f} мс")
    objects, bg = background_normalization(common.salt_pepper(common.demo_objects()[0], 0.03))
    if truth is not None:
        print(f"нормализация фона: IoU {common.iou(objects > 40, truth):.3f}")
    common.grid([(noisy, "шум 10%"), (median(noisy, k), "медиана 5x5"), (minimax(noisy, 3), "минимакс 3x3"),
                 (bg, "карта фона"), (objects, "объекты = изображение - фон")], args.out, "lab06.png")


if __name__ == "__main__":
    main()
