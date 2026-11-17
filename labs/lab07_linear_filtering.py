"""Лабораторная работа 7. Линейная фильтрация в пространственной области.

Дискретная свёртка: (I * K)(x, y) = SUM_i SUM_j K(i, j) I(x - i, y - j).
cv2.filter2D считает КОРРЕЛЯЦИЮ (ядро не переворачивается); для симметричных
ядер результаты совпадают, для Собеля отличаются знаком.
Сложность прямой свёртки O(H W kh kw); для разделимых ядер (Гаусс) --
O(H W (kh + kw)).

Запуск: python lab07_linear_filtering.py --demo | --image файл.png
"""

import sys

import cv2
import numpy as np

import common


# --------------------------------------------------------------------------
# Ядра
# --------------------------------------------------------------------------
def box_kernel(k):
    """Усредняющее ядро k x k: все веса 1/k^2 (сумма = 1, яркость сохраняется)."""
    return np.full((k, k), 1.0 / (k * k))


def gauss_kernel_1d(k, sigma):
    """Одномерное гауссово ядро длины k: g(x) ~ exp(-x^2 / (2 sigma^2)), сумма = 1."""
    x = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def gauss_kernel(k, sigma):
    """Двумерное гауссово ядро как внешнее произведение двух 1D-ядер (разделимость)."""
    g = gauss_kernel_1d(k, sigma)
    return np.outer(g, g)


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T
LAPLACE = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)  # I - Laplace(I)


# --------------------------------------------------------------------------
# Свёртка / корреляция
# --------------------------------------------------------------------------
def correlate2d(gray, kernel):
    """Корреляция с ядром (как cv2.filter2D), выход float64 без округления.

    Для каждого смещения (dy, dx) внутри ядра складываем весь сдвинутый кадр
    с весом K[dy, dx]: цикл по ядру (kh*kw итераций), а не по пикселям.
    Края: зеркальное дополнение (reflect = BORDER_REFLECT_101).
    """
    common.check_gray_u8(gray)
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Размер ядра должен быть нечётным")
    h, w = gray.shape
    padded = np.pad(gray.astype(np.float64), ((kh // 2, kh // 2), (kw // 2, kw // 2)),
                    mode="reflect")
    out = np.zeros((h, w))
    for dy in range(kh):
        for dx in range(kw):
            out += kernel[dy, dx] * padded[dy:dy + h, dx:dx + w]
    return out


def convolve2d(gray, kernel):
    """Истинная свёртка = корреляция с ядром, повёрнутым на 180 градусов."""
    return correlate2d(gray, np.asarray(kernel)[::-1, ::-1])


def separable_filter(gray, kx, ky):
    """Разделимая фильтрация: сначала строки ядром kx, затем столбцы ядром ky."""
    row_k = np.asarray(kx, dtype=np.float64).reshape(1, -1)
    col_k = np.asarray(ky, dtype=np.float64).reshape(-1, 1)
    tmp = correlate2d(gray, row_k)
    # Второй проход должен идти по float-результату; делаем его тем же приёмом.
    h, w = tmp.shape
    padded = np.pad(tmp, ((col_k.shape[0] // 2,) * 2, (0, 0)), mode="reflect")
    out = np.zeros((h, w))
    for dy in range(col_k.shape[0]):
        out += col_k[dy, 0] * padded[dy:dy + h, :]
    return out


def to_u8(x):
    """Округление и насыщение float -> uint8 (clip до [0, 255])."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def main():
    """Точка входа."""
    args = common.build_parser(__doc__).parse_args()
    if not (args.demo or args.image):
        print("Нужен --demo или --image")
        return 1
    gray = common.to_gray(common.get_input(args, lambda: common.demo_objects(noise=0.0)[0]))
    common.ensure_dir(args.out)

    # Сравнение с cv2.filter2D (корреляция) и cv2.GaussianBlur.
    k = 5
    box = correlate2d(gray, box_kernel(k))
    ref = cv2.filter2D(gray.astype(np.float32), -1, box_kernel(k).astype(np.float32),
                       borderType=cv2.BORDER_REFLECT_101)
    print(f"[box {k}x{k}] max|ours - cv2.filter2D| = {np.abs(box - ref).max():.2e}")

    gx = correlate2d(gray, SOBEL_X)
    gx_cv = cv2.filter2D(gray.astype(np.float32), -1, SOBEL_X.astype(np.float32),
                         borderType=cv2.BORDER_REFLECT_101)
    print(f"[Sobel X] корреляция vs filter2D: {np.abs(gx - gx_cv).max():.2e} | "
          f"свёртка отличается знаком: max|conv + corr| = "
          f"{np.abs(convolve2d(gray, SOBEL_X) + gx).max():.2e}")

    g = gauss_kernel(k, 1.0)
    gauss = correlate2d(gray, g)
    gauss_sep = separable_filter(gray, gauss_kernel_1d(k, 1.0), gauss_kernel_1d(k, 1.0))
    gauss_cv = cv2.GaussianBlur(gray.astype(np.float32), (k, k), 1.0,
                                borderType=cv2.BORDER_REFLECT_101)
    print(f"[Гаусс] разделимо == 2D: max|diff| = {np.abs(gauss - gauss_sep).max():.2e}; "
          f"vs cv2.GaussianBlur: {np.abs(gauss - gauss_cv).max():.2e}")

    # Замер времени: 2D против разделимого против OpenCV.
    for kk in (3, 7, 15):
        sigma = kk / 5.0
        t2 = common.timeit(correlate2d, gray, gauss_kernel(kk, sigma), repeat=3)
        ts = common.timeit(separable_filter, gray, gauss_kernel_1d(kk, sigma),
                           gauss_kernel_1d(kk, sigma), repeat=3)
        tc = common.timeit(cv2.GaussianBlur, gray, (kk, kk), sigma, repeat=3)
        print(f"[время] k={kk:2d}: 2D {t2 * 1e3:6.1f} мс | разделимо {ts * 1e3:6.1f} мс | "
              f"cv2 {tc * 1e3:.3f} мс")

    # Шум: на гауссовом шуме Гаусс против усреднения по PSNR и по размытию границ.
    noisy = common.add_gaussian_noise(gray, 20.0)
    print(f"[шум s=20] PSNR: шум {common.psnr(gray, noisy):.1f} | "
          f"box5 {common.psnr(gray, to_u8(correlate2d(noisy, box_kernel(5)))):.1f} | "
          f"gauss5 {common.psnr(gray, to_u8(correlate2d(noisy, gauss_kernel(5, 1.0)))):.1f} дБ")

    sharp = to_u8(correlate2d(gray, SHARPEN))
    lap = np.abs(correlate2d(gray, LAPLACE))
    mag = np.hypot(gx, correlate2d(gray, SOBEL_Y))
    common.save_grid([(gray, "исходное"), (to_u8(box), "усреднение 5x5"),
                      (to_u8(gauss), "Гаусс 5x5, s=1"), (sharp, "резкость"),
                      (to_u8(mag / mag.max() * 255), "Собель |grad|"),
                      (to_u8(lap / lap.max() * 255), "Лапласиан |.|")],
                     f"{args.out}/lab07_linear.png", ncols=3)
    print(f"Графики: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
