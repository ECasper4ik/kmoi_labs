"""Лабораторная работа 8. Преобразование Хафа.

Прямая в нормальной параметризации: rho = x cos(theta) + y sin(theta).
Каждая точка контура голосует за все прямые, проходящие через неё
(синусоида в плоскости (theta, rho)); пики аккумулятора -- найденные прямые.
Параметры (rho, theta), а не (k, b) из y = kx + b: у вертикальной прямой
k = бесконечность, а theta ограничен диапазоном [0, 180).
Окружность известного радиуса: центр (a, b) = (x - r cos t, y - r sin t).

Запуск: python lab08_hough.py --demo | --image файл.png
"""

import sys

import cv2
import numpy as np

import common


def hough_lines(edges, theta_step=1.0, rho_step=1.0):
    """Аккумулятор Хафа для прямых.

    Args:
        edges: бинарная карта контуров (ненулевые пиксели голосуют).
        theta_step: шаг по углу в градусах (меньше шаг -- точнее и дольше).
        rho_step: шаг по rho в пикселях.

    Returns:
        (acc[rho, theta] int32, rhos, thetas в радианах)
    """
    common.check_gray_u8(edges)
    ys, xs = np.nonzero(edges)
    h, w = edges.shape
    diag = int(np.ceil(np.hypot(h, w)))             # |rho| <= диагональ изображения
    thetas = np.deg2rad(np.arange(0.0, 180.0, theta_step))
    rhos = np.arange(-diag, diag + 1, rho_step)
    acc = np.zeros((rhos.size, thetas.size), np.int32)
    for j, theta in enumerate(thetas):
        # Все точки контура голосуют за свою rho при данном theta одним вызовом bincount.
        rho = xs * np.cos(theta) + ys * np.sin(theta)
        idx = np.rint((rho + diag) / rho_step).astype(np.int64)
        acc[:, j] = np.bincount(idx, minlength=rhos.size)[:rhos.size]
    return acc, rhos, thetas


def find_peaks(acc, threshold, nms=9, max_peaks=10):
    """Локальные максимумы аккумулятора выше порога (подавление немаксимумов nms x nms).

    Без подавления одна прямая даёт «облако» соседних пиков.
    """
    acc_f = acc.astype(np.float32)
    local_max = cv2.dilate(acc_f, np.ones((nms, nms), np.uint8))
    peaks = np.argwhere((acc_f == local_max) & (acc_f >= threshold))
    order = np.argsort(-acc_f[peaks[:, 0], peaks[:, 1]])
    return peaks[order][:max_peaks]


def hough_circles(edges, r_min, r_max, min_fraction=0.6):
    """Окружности по Хафу: для каждого радиуса голосуем за центры.

    Для радиуса r каждая точка контура даёт окружность возможных центров.
    Оценка = голоса в окне 3x3 / число углов: ~1 для идеального контура
    толщиной в 1 пиксель (толстый или двойной контур даёт больше), поэтому
    порог min_fraction сопоставим для разных r.

    Returns:
        список (score, x_center, y_center, r), лучшие первыми, без дублей.
    """
    common.check_gray_u8(edges)
    ys, xs = np.nonzero(edges)
    h, w = edges.shape
    found = []
    for r in range(r_min, r_max + 1):
        n_angles = max(36, int(2 * np.pi * r))
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        acc = np.zeros(h * w)
        for t in angles:
            a = np.rint(xs - r * np.cos(t)).astype(int)
            b = np.rint(ys - r * np.sin(t)).astype(int)
            ok = (a >= 0) & (a < w) & (b >= 0) & (b < h)
            acc += np.bincount(b[ok] * w + a[ok], minlength=h * w)
        # Сумма голосов в окне 3x3: линия контура имеет толщину ~1 пикселя, и голоса
        # одной окружности попадают в соседние ячейки аккумулятора.
        acc = cv2.boxFilter(acc.reshape(h, w).astype(np.float32), -1, (3, 3), normalize=False)
        score = acc.max() / n_angles
        by, bx = np.unravel_index(np.argmax(acc), acc.shape)
        if score >= min_fraction:
            found.append((float(score), int(bx), int(by), r))
    found.sort(reverse=True)
    result = []
    for cand in found:
        if all(np.hypot(cand[1] - c[1], cand[2] - c[2]) > max(cand[3], c[3])
               for c in result):
            result.append(cand)
    return result


def lines_to_points(rho, theta, size):
    """Две точки на достаточном удалении для рисования прямой (rho, theta)."""
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    d = float(np.hypot(*size))
    return ((int(x0 - d * b), int(y0 + d * a)), (int(x0 + d * b), int(y0 - d * a)))


def main():
    """Точка входа."""
    args = common.build_parser(__doc__).parse_args()
    if not (args.demo or args.image):
        print("Нужен --demo или --image")
        return 1
    gray = common.to_gray(common.get_input(args, common.demo_lines))
    common.ensure_dir(args.out)
    h, w = gray.shape

    edges = cv2.Canny(gray, 50, 150)  # подготовка контуров; сам Хаф -- ручной
    acc, rhos, thetas = hough_lines(edges)
    peaks = find_peaks(acc, threshold=0.5 * acc.max(), max_peaks=6)
    print("[наш Хаф] (rho, theta°, голоса):")
    for ri, ti in peaks:
        print(f"   {rhos[ri]:7.1f} {np.rad2deg(thetas[ti]):6.1f} {acc[ri, ti]:5d}")
    ref = cv2.HoughLines(edges, 1, np.pi / 180, int(0.5 * acc.max()))
    print("[cv2.HoughLines] первые линии (rho, theta°):",
          [(round(float(r), 1), round(float(np.rad2deg(t)), 1)) for r, t in ref[:6, 0]]
          if ref is not None else "нет")

    circles = hough_circles(edges, 25, 45)[:2]
    print(f"[наш Хаф, окружности] (score, x, y, r): {[(round(c[0], 2), *c[1:]) for c in circles]}")

    for step in (2.0, 1.0, 0.5):
        print(f"[время] theta_step={step}: "
              f"{common.timeit(hough_lines, edges, step, repeat=2) * 1e3:.1f} мс")

    # Тест на шум: сильный гауссов шум, размытие перед Canny.
    noisy = common.add_gaussian_noise(gray, 25.0)
    edges_n = cv2.Canny(cv2.GaussianBlur(noisy, (5, 5), 1.4), 50, 150)
    acc_n, _, _ = hough_lines(edges_n)
    peaks_n = find_peaks(acc_n, threshold=0.5 * acc_n.max(), max_peaks=6)
    print(f"[шум s=25] найдено пиков: {len(peaks_n)} (без шума {len(peaks)}), "
          f"контурных точек: {int((edges_n > 0).sum())} против {int((edges > 0).sum())}")

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for ri, ti in peaks:
        p, q = lines_to_points(rhos[ri], thetas[ti], (h, w))
        cv2.line(vis, p, q, (0, 0, 255), 1)
    for _, cx, cy, r in circles:
        cv2.circle(vis, (cx, cy), r, (0, 255, 0), 1)
    acc_view = (acc / max(acc.max(), 1) * 255).astype(np.uint8)
    common.save_grid([(gray, "исходное"), (edges, "контуры (Canny)"),
                      (acc_view, "аккумулятор (rho x theta)"), (vis, "найденные прямые/окружности"),
                      (edges_n, "контуры при шуме s=25")], f"{args.out}/lab08_hough.png", ncols=3)
    print(f"Графики: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
