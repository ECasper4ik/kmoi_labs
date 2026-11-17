"""Лабораторная работа 11. Выделение контуров на полутоновых изображениях.

Контур -- место резкого перепада яркости, то есть максимума модуля градиента
(первая производная) или нуля второй производной (Лапласиан).
Градиент: G = (gx, gy), |G| = sqrt(gx^2 + gy^2), направление atan2(gy, gx).
Операторы Робертса (2x2), Прюитт и Собеля (3x3, со сглаживанием поперёк).
Схема Канни: сглаживание -> градиент -> подавление немаксимумов -> гистерезис.

Запуск: python lab11_gray_contours.py --demo | --image файл.png
"""

import sys

import cv2
import numpy as np

import common
from lab07_linear_filtering import (SOBEL_X, SOBEL_Y, correlate2d, gauss_kernel)

PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
PREWITT_Y = PREWITT_X.T


def roberts(gray):
    """Оператор Робертса: диагональные разности 2x2.

    gx = I(x+1, y+1) - I(x, y),  gy = I(x+1, y) - I(x, y+1).
    Минимальная апертура: быстро, но очень чувствителен к шуму.
    """
    common.check_gray_u8(gray)
    g = np.pad(gray.astype(np.float64), ((0, 1), (0, 1)), mode="edge")
    gx = g[1:, 1:] - g[:-1, :-1]
    gy = g[:-1, 1:] - g[1:, :-1]
    return gx, gy


def gradient(gray, kind="sobel"):
    """Компоненты градиента (gx, gy) выбранным оператором."""
    if kind == "roberts":
        return roberts(gray)
    kx, ky = {"sobel": (SOBEL_X, SOBEL_Y), "prewitt": (PREWITT_X, PREWITT_Y)}[kind]
    return correlate2d(gray, kx), correlate2d(gray, ky)


def magnitude(gx, gy, norm="l2"):
    """Модуль градиента: L2 = sqrt(gx^2 + gy^2); L1 = |gx| + |gy| (быстрее, без корня)."""
    return np.hypot(gx, gy) if norm == "l2" else np.abs(gx) + np.abs(gy)


def non_max_suppression(mag, gx, gy):
    """Подавление немаксимумов: оставить пиксель, если он максимум вдоль градиента.

    Направление градиента квантуется до 4 направлений (0, 45, 90, 135 градусов);
    пиксель сравнивается с двумя соседями вдоль этого направления.
    """
    h, w = mag.shape
    angle = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    q = ((angle + 22.5) // 45).astype(int) % 4
    m = np.pad(mag, 1, mode="constant")
    # (dy, dx) соседей по каждому квантованному направлению (ось y направлена вниз).
    offsets = {0: ((0, 1), (0, -1)), 1: ((1, 1), (-1, -1)),
               2: ((1, 0), (-1, 0)), 3: ((1, -1), (-1, 1))}
    keep = np.zeros((h, w), bool)
    for d, ((ay, ax), (by, bx)) in offsets.items():
        n1 = m[1 + ay:1 + ay + h, 1 + ax:1 + ax + w]
        n2 = m[1 + by:1 + by + h, 1 + bx:1 + bx + w]
        keep |= (q == d) & (mag >= n1) & (mag >= n2)
    return np.where(keep, mag, 0.0)


def hysteresis(nms, low, high):
    """Гистерезисная пороговая обработка: слабые пиксели (>= low) остаются,
    только если связаны с сильным (>= high). Связность компонент -- через cv2."""
    strong = nms >= high
    weak = (nms >= low).astype(np.uint8)
    _, comp = cv2.connectedComponents(weak, connectivity=8)
    keep_ids = np.unique(comp[strong])
    keep_ids = keep_ids[keep_ids > 0]
    return (np.isin(comp, keep_ids)).astype(np.uint8) * 255


def canny_manual(gray, sigma=1.4, low_frac=0.1, high_frac=0.25):
    """Схема Канни: Гаусс -> Собель -> NMS -> гистерезис (пороги -- доли максимума)."""
    k = 2 * int(np.ceil(3 * sigma)) + 1
    smooth = correlate2d(gray, gauss_kernel(k, sigma))
    smooth_u8 = np.clip(np.rint(smooth), 0, 255).astype(np.uint8)
    gx, gy = gradient(smooth_u8, "sobel")
    nms = non_max_suppression(magnitude(gx, gy), gx, gy)
    return hysteresis(nms, low_frac * nms.max(), high_frac * nms.max())


def log_zero_crossing(gray, sigma=2.0, thr_frac=0.05):
    """Контуры по нулям лапласиана гауссиана (LoG).

    LoG(x, y) = -1/(pi s^4) * (1 - r^2 / (2 s^2)) * exp(-r^2 / (2 s^2)).
    Контур -- смена знака отклика между соседями при достаточном размахе.
    """
    k = 2 * int(np.ceil(3 * sigma)) + 1
    ax = np.arange(k) - k // 2
    xx, yy = np.meshgrid(ax, ax)
    r2 = xx ** 2 + yy ** 2
    kernel = -1.0 / (np.pi * sigma ** 4) * (1 - r2 / (2 * sigma ** 2)) * np.exp(-r2 / (2 * sigma ** 2))
    kernel -= kernel.mean()  # сумма весов = 0: на плоском участке отклик ровно 0
    resp = correlate2d(gray, kernel)
    thr = thr_frac * np.abs(resp).max()
    edges = np.zeros(gray.shape, bool)
    for axis in (0, 1):
        a = resp
        b = np.roll(resp, -1, axis=axis)
        crossing = (a * b < 0) & (np.abs(a - b) > thr)
        edges |= crossing
    return edges.astype(np.uint8) * 255


def main():
    """Точка входа."""
    args = common.build_parser(__doc__).parse_args()
    if not (args.demo or args.image):
        print("Нужен --demo или --image")
        return 1
    if args.image:
        gray, truth = common.to_gray(common.load_image(args.image)), None
    else:
        gray, truth = common.demo_objects(noise=6.0)
    common.ensure_dir(args.out)

    gx, gy = gradient(gray, "sobel")
    mag = magnitude(gx, gy)
    ref_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    print(f"[Собель] max|gx - cv2.Sobel| = {np.abs(gx - ref_x).max():.2e}")

    ours = canny_manual(gray)
    ref_canny = cv2.Canny(gray, 50, 150)
    print(f"[Канни] наш {int((ours > 0).sum())} пикселей | cv2.Canny {int((ref_canny > 0).sum())} | "
          f"взаимное F1 (допуск 1 px) = {common.f1_score(ref_canny, ours, 1):.3f}")

    reference = None
    if truth is not None:
        t = truth.astype(np.uint8)
        reference = (t - cv2.erode(t, np.ones((3, 3), np.uint8))) * 255  # граница эталона
        simple = (mag > 0.3 * mag.max()).astype(np.uint8) * 255
        print(f"[F1 к эталонной границе, допуск 2 px] порог Собеля {common.f1_score(reference, simple, 2):.3f} | "
              f"наш Канни {common.f1_score(reference, ours, 2):.3f} | "
              f"LoG {common.f1_score(reference, log_zero_crossing(gray), 2):.3f}")

    for name in ("roberts", "prewitt", "sobel"):
        print(f"[время] {name:8s}: {common.timeit(gradient, gray, name) * 1e3:.2f} мс")
    print(f"[время] canny_manual {common.timeit(canny_manual, gray, repeat=2) * 1e3:.1f} мс | "
          f"cv2.Canny {common.timeit(cv2.Canny, gray, 50, 150) * 1e3:.3f} мс")

    # Тест на шум: градиент без сглаживания против Канни со сглаживанием.
    noisy = common.add_gaussian_noise(gray, 20.0)
    gnx, gny = gradient(noisy, "roberts")
    rob = (magnitude(gnx, gny) > 0.3 * magnitude(gnx, gny).max()).astype(np.uint8) * 255
    if reference is not None:
        n_sobel = magnitude(*gradient(noisy, "sobel"))
        print(f"[шум s=20, F1] Робертс {common.f1_score(reference, rob, 2):.3f} | "
              f"Собель {common.f1_score(reference, (n_sobel > 0.3 * n_sobel.max()).astype(np.uint8) * 255, 2):.3f} | "
              f"Канни {common.f1_score(reference, canny_manual(noisy, sigma=2.0), 2):.3f}")

    common.save_grid([(gray, "исходное"), (np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8), "|grad| Собеля"),
                      (ours, "Канни (наш)"), (ref_canny, "cv2.Canny"),
                      (log_zero_crossing(gray), "LoG: нули"), (canny_manual(noisy, sigma=2.0), "Канни при шуме s=20")],
                     f"{args.out}/lab11_contours.png", ncols=3)
    print(f"Графики: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
