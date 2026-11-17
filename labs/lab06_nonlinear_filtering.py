"""Лабораторная работа 6. Методы полутоновой нелинейной фильтрации.

Ранговые фильтры: в окне k x k значения упорядочиваются, на выход идёт
элемент заданного ранга: min (ранг 0), медиана (ранг k^2/2), max (ранг k^2-1).
Медиана убирает импульсный шум и сохраняет перепады. Минимаксный фильтр
(min, затем max) убирает светлые выбросы за два прохода. Вычислительная
сложность ранговой статистики растёт сверхлинейно с размером апертуры.

Запуск: python lab06_nonlinear_filtering.py --demo | --image файл.png
"""

import sys

import cv2
import numpy as np

import common


def _check_k(k):
    """Апертура должна быть нечётной и >= 1."""
    if k < 1 or k % 2 == 0:
        raise ValueError("Размер апертуры k должен быть нечётным >= 1")


def rank_filter(gray, k, rank):
    """Ранговый фильтр: элемент с номером `rank` (0..k*k-1) в упорядоченном окне.

    Краевые пиксели: зеркальное дополнение (reflect), иначе окно вышло бы
    за границу изображения.
    """
    common.check_gray_u8(gray)
    _check_k(k)
    if not 0 <= rank < k * k:
        raise ValueError("rank вне диапазона 0..k*k-1")
    windows = common.sliding(gray, k)                   # (H, W, k, k), без копирования
    flat = windows.reshape(gray.shape[0], gray.shape[1], k * k)
    # np.partition: O(k^2) в среднем вместо полной сортировки O(k^2 log k^2).
    return np.partition(flat, rank, axis=-1)[..., rank]


def median_filter(gray, k):
    """Медианный фильтр (ранг k*k//2)."""
    return rank_filter(gray, k, (k * k) // 2)


def _extreme_separable(gray, k, func):
    """min/max по окну k x k как два 1D-прохода: min(k x k) = min по строкам от min по столбцам.

    Для min и max это точное тождество (для медианы -- нет), и сложность
    O(H W k) вместо O(H W k^2): окно 35 x 35 перестаёт быть проблемой.
    """
    common.check_gray_u8(gray)
    _check_k(k)
    r = k // 2
    tmp = func(np.lib.stride_tricks.sliding_window_view(
        np.pad(gray, ((0, 0), (r, r)), mode="reflect"), k, axis=1), axis=-1)
    return func(np.lib.stride_tricks.sliding_window_view(
        np.pad(tmp, ((r, r), (0, 0)), mode="reflect"), k, axis=0), axis=-1)


def min_filter(gray, k):
    """Фильтр минимума (эрозия полутонового изображения плоским элементом)."""
    return _extreme_separable(gray, k, np.min)


def max_filter(gray, k):
    """Фильтр максимума (дилатация)."""
    return _extreme_separable(gray, k, np.max)


def minimax_filter(gray, k):
    """Минимаксный фильтр: min, затем max. Убирает светлые выбросы меньше окна."""
    return max_filter(min_filter(gray, k), k)


def maximin_filter(gray, k):
    """Максиминный фильтр: max, затем min. Убирает тёмные выбросы."""
    return min_filter(max_filter(gray, k), k)


def background_normalization(gray, k_small=3, k_big=35, bright_objects=True):
    """Нормализация фона (методичка): малая апертура убирает шум, большая --
    объекты, оставляя фон; вычитание карты фона выделяет объекты.

    Для светлых объектов фон оценивается минимаксным фильтром большой
    апертуры (он «срезает» светлое меньше окна).
    """
    smooth = median_filter(gray, k_small)
    if bright_objects:
        background = minimax_filter(smooth, k_big)
        diff = smooth.astype(np.int16) - background.astype(np.int16)
    else:
        background = maximin_filter(smooth, k_big)
        diff = background.astype(np.int16) - smooth.astype(np.int16)
    return np.clip(diff, 0, 255).astype(np.uint8), background


def main():
    """Точка входа."""
    args = common.build_parser(__doc__).parse_args()
    if not (args.demo or args.image):
        print("Нужен --demo или --image")
        return 1
    if args.image:
        gray, truth = common.to_gray(common.load_image(args.image)), None
    else:
        gray, truth = common.demo_objects(noise=0.0)
    common.ensure_dir(args.out)

    # Сравнение с OpenCV в «внутренней» области (границы обрабатываются по-разному).
    k = 5
    med = median_filter(gray, k)
    ref = cv2.medianBlur(gray, k)
    inner = (slice(k, -k), slice(k, -k))
    print(f"[медиана {k}x{k}] max|diff| внутри = "
          f"{np.abs(med[inner].astype(int) - ref[inner].astype(int)).max()}")
    kern = np.ones((k, k), np.uint8)
    print(f"[min] max|diff| внутри = "
          f"{np.abs(min_filter(gray, k)[inner].astype(int) - cv2.erode(gray, kern)[inner].astype(int)).max()}"
          f" | [max] = "
          f"{np.abs(max_filter(gray, k)[inner].astype(int) - cv2.dilate(gray, kern)[inner].astype(int)).max()}")

    # Тест на шум: импульсный 10%, медиана против среднего.
    noisy = common.add_salt_pepper(gray, 0.10)
    box = cv2.blur(noisy, (k, k))
    print(f"[шум s&p 10%] PSNR: зашумлённое {common.psnr(gray, noisy):.1f} | "
          f"среднее {common.psnr(gray, box):.1f} | медиана {common.psnr(gray, median_filter(noisy, k)):.1f} дБ")

    # Зависимость времени от апертуры (сверхлинейный рост).
    for kk in (3, 5, 9, 15):
        t = common.timeit(median_filter, gray, kk, repeat=2)
        print(f"[время] медиана {kk:2d}x{kk:<2d}: {t * 1e3:8.1f} мс | cv2.medianBlur: "
              f"{common.timeit(cv2.medianBlur, gray, kk) * 1e3:.2f} мс | "
              f"min (разделимый): {common.timeit(min_filter, gray, kk, repeat=2) * 1e3:.2f} мс")

    # Нормализация фона на объектах с градиентом освещения.
    img_bg, _ = common.demo_objects(noise=8.0)
    objects, background = background_normalization(common.add_salt_pepper(img_bg, 0.03))
    if truth is not None:
        print(f"[нормализация фона] IoU (порог 40) = "
              f"{common.iou(objects > 40, truth):.3f}")

    common.save_grid([(noisy, "импульсный шум 10%"), (box, "усреднение"),
                      (median_filter(noisy, k), "медиана 5x5"),
                      (minimax_filter(noisy, 3), "минимакс 3x3"),
                      (background, "карта фона"), (objects, "объекты (img - фон)")],
                     f"{args.out}/lab06_nonlinear.png", ncols=3)
    print(f"Графики: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
