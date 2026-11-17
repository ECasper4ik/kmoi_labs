"""Лабораторная работа 3. Профили яркости.

Профиль -- график яркости вдоль выбранной линии: P(t) = I(x0 + t*dx, y0 + t*dy).
Вдоль произвольной линии координаты нецелые, поэтому яркость считается
билинейной интерполяцией. Усреднение по ширине линии подавляет шум в sqrt(w) раз.

Запуск: python lab03_profiles.py --demo | --image файл.png
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

import common


def bilinear(gray, xs, ys):
    """Билинейная интерполяция яркости в точках (xs, ys) (float).

    I(x, y) = (1-wy) * [(1-wx) I00 + wx I10] + wy * [(1-wx) I01 + wx I11],
    где wx, wy -- дробные части координат. Вне изображения координаты
    зажимаются к границе (аналог BORDER_REPLICATE).
    """
    h, w = gray.shape
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)  # на правом/нижнем крае второй сосед = сам пиксель
    y1 = np.minimum(y0 + 1, h - 1)
    wx = xs - x0
    wy = ys - y0
    g = gray.astype(np.float64)
    top = (1 - wx) * g[y0, x0] + wx * g[y0, x1]
    bottom = (1 - wx) * g[y1, x0] + wx * g[y1, x1]
    return (1 - wy) * top + wy * bottom


def profile_line(gray, p0, p1, width=1):
    """Профиль вдоль отрезка p0 -> p1 (точки в виде (x, y)).

    Args:
        gray: 2D изображение.
        p0, p1: концы отрезка.
        width: ширина полосы; значения усредняются по перпендикуляру.

    Returns:
        (profile float64, xs, ys) -- профиль и координаты его отсчётов.
    """
    common.check_gray_u8(gray)
    (x0, y0), (x1, y1) = p0, p1
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length == 0:
        raise ValueError("Начало и конец отрезка совпадают")
    n = int(np.ceil(length)) + 1  # примерно один отсчёт на пиксель
    t = np.linspace(0.0, 1.0, n)
    xs = x0 + t * (x1 - x0)
    ys = y0 + t * (y1 - y0)
    nx, ny = -(y1 - y0) / length, (x1 - x0) / length  # единичная нормаль
    offsets = np.linspace(-(width - 1) / 2.0, (width - 1) / 2.0, width)
    acc = np.zeros(n)
    for off in offsets:
        acc += bilinear(gray, xs + off * nx, ys + off * ny)
    return acc / width, xs, ys


def row_profile(gray, y):
    """Горизонтальный профиль: строка с номером y."""
    common.check_gray_u8(gray)
    return gray[y, :].astype(np.float64)


def col_profile(gray, x):
    """Вертикальный профиль: столбец с номером x."""
    common.check_gray_u8(gray)
    return gray[:, x].astype(np.float64)


def edge_positions(profile, thr):
    """Индексы, где модуль первой разности профиля превышает порог thr.

    Производная профиля (конечная разность P[i+1] - P[i]) максимальна на
    перепаде яркости -- так находят границы объектов.
    """
    d = np.diff(profile)
    return np.flatnonzero(np.abs(d) > thr)


def reference_profile(gray, xs, ys):
    """Эталон: тот же профиль через cv2.remap (INTER_LINEAR)."""
    return cv2.remap(gray.astype(np.float32), xs.astype(np.float32).reshape(1, -1),
                     ys.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).ravel()


def main():
    """Точка входа."""
    args = common.build_parser(__doc__).parse_args()
    if not (args.demo or args.image):
        print("Нужен --demo или --image")
        return 1
    gray = common.to_gray(common.get_input(args, lambda: common.demo_objects()[0]))
    h, w = gray.shape
    common.ensure_dir(args.out)

    p0, p1 = (5, h // 4), (w - 6, h // 2)  # наклонная линия через несколько объектов
    prof, xs, ys = profile_line(gray, p0, p1, width=1)
    ref = reference_profile(gray, xs, ys)
    print(f"[сравнение] max|ours - cv2.remap| = {np.abs(prof - ref).max():.4f}")

    # Тест на шум: ширина полосы 1 против 9 на зашумлённом изображении.
    noisy = common.add_gaussian_noise(gray, 15.0)
    p_n1, _, _ = profile_line(noisy, p0, p1, width=1)
    p_n9, _, _ = profile_line(noisy, p0, p1, width=9)
    clean9, _, _ = profile_line(gray, p0, p1, width=9)  # честная база: тот же width без шума
    print(f"[шум] RMS отклонения от чистого профиля той же ширины: width=1: "
          f"{np.sqrt(np.mean((p_n1 - prof) ** 2)):.1f}, "
          f"width=9: {np.sqrt(np.mean((p_n9 - clean9) ** 2)):.1f}")

    print(f"[время] profile_line {common.timeit(profile_line, gray, p0, p1) * 1e3:.3f} мс")
    print(f"[границы] индексы перепадов (thr=20): {edge_positions(prof, 20)[:10]} ...")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].plot([p0[0], p1[0]], [p0[1], p1[1]], "r-")
    axes[0, 0].set_title("изображение и линия профиля")
    axes[0, 1].plot(prof, "k", label="ours")
    axes[0, 1].plot(ref, "r--", label="cv2.remap")
    axes[0, 1].legend()
    axes[0, 1].set_title("профиль (билинейная интерполяция)")
    axes[1, 0].plot(row_profile(gray, h // 2), "k")
    axes[1, 0].set_title("горизонтальный профиль (строка)")
    axes[1, 1].plot(p_n1, "gray", label="width=1")
    axes[1, 1].plot(p_n9, "k", label="width=9")
    axes[1, 1].legend()
    axes[1, 1].set_title("шум: усреднение по ширине")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "lab03_profiles.png"), dpi=110)
    plt.close(fig)
    print(f"Графики: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
