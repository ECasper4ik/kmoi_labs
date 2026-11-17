"""Лабораторная работа 4. Проекции.

Горизонтальная проекция: PH(y) = SUM_x I(x, y); вертикальная: PV(x) = SUM_y I(x, y).
Для бинарного изображения это число пикселей объекта в строке/столбце.
Применение: выделение строк текста (впадины PH), оценка угла наклона текста
(проекция «острее» всего при правильном угле).

Запуск: python lab04_projections.py --demo | --image файл.png
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

import common
from lab05_binarization import global_threshold, otsu_threshold


def projections(binary):
    """Вернуть (PH, PV) бинарного изображения (объект = 1, фон = 0).

    Суммирование в int64: сумма uint8 по строке переполнила бы uint8.
    """
    common.check_gray_u8(binary)
    obj = (binary > 0).astype(np.int64)
    return obj.sum(axis=1), obj.sum(axis=0)


def find_runs(profile, thr=0, min_len=1):
    """Непрерывные участки, где profile > thr. Вернуть список (start, end) (end не включая)."""
    mask = np.concatenate(([0], (profile > thr).astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(mask))  # +1 -> начало серии, -1 -> конец
    runs = list(zip(edges[::2], edges[1::2]))
    return [(s, e) for s, e in runs if e - s >= min_len]


def rotate(img, angle):
    """Поворот на angle градусов вокруг центра (билинейная интерполяция)."""
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def estimate_skew(binary, max_angle=10.0, step=0.5):
    """Оценка угла наклона текста: максимум дисперсии горизонтальной проекции.

    При правильном угле строки текста выровнены -> PH имеет резкие пики и
    глубокие впадины -> дисперсия максимальна.
    """
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + step, step):
        ph, _ = projections(rotate(binary, angle))
        score = float(np.var(ph))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def main():
    """Точка входа."""
    args = common.build_parser(__doc__).parse_args()
    if not (args.demo or args.image):
        print("Нужен --demo или --image")
        return 1
    gray = common.to_gray(common.get_input(args, lambda: common.demo_text(angle=3.0)))
    common.ensure_dir(args.out)

    # Объект тёмный на светлом -> инвертируем бинарку, чтобы объект был «1».
    binary = 255 - global_threshold(gray, otsu_threshold(gray))
    ph, pv = projections(binary)

    ref_h = cv2.reduce((binary > 0).astype(np.float64), 1, cv2.REDUCE_SUM).ravel()
    ref_v = cv2.reduce((binary > 0).astype(np.float64), 0, cv2.REDUCE_SUM).ravel()
    print(f"[сравнение] max|PH - cv2.reduce| = {np.abs(ph - ref_h).max()}, "
          f"max|PV - cv2.reduce| = {np.abs(pv - ref_v).max()}")

    angle = estimate_skew(binary)
    print(f"[наклон] угол коррекции: {angle:+.1f} град. (демо повёрнуто на +3.0, "
          "значит для выравнивания нужен поворот на -3.0)")
    fixed = rotate(binary, angle)
    ph_fixed, _ = projections(fixed)
    lines = find_runs(ph_fixed, thr=0.02 * ph_fixed.max(), min_len=5)
    print(f"[строки текста] найдено {len(lines)}: {[(int(a), int(b)) for a, b in lines]}")

    print(f"[время] projections {common.timeit(projections, binary) * 1e3:.3f} мс | "
          f"estimate_skew {common.timeit(estimate_skew, binary, repeat=1):.2f} с")

    # Шум: проекция по неочищенной бинарке vs после медианы.
    noisy = common.add_salt_pepper(binary, 0.02)
    ph_n, _ = projections(noisy)
    ph_f, _ = projections(cv2.medianBlur(noisy, 3))
    print(f"[шум 2%] RMS(PH_noisy - PH) = {np.sqrt(np.mean((ph_n - ph) ** 2)):.1f}, "
          f"после медианы {np.sqrt(np.mean((ph_f - ph) ** 2)):.1f}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].imshow(binary, cmap="gray")
    axes[0, 0].set_title("бинарное (объект = белый)")
    axes[0, 1].imshow(fixed, cmap="gray")
    axes[0, 1].set_title(f"после компенсации угла {angle:+.1f}")
    axes[1, 0].plot(ph, "k", label="до")
    axes[1, 0].plot(ph_fixed, "r", label="после")
    axes[1, 0].legend()
    axes[1, 0].set_title("горизонтальная проекция PH(y)")
    axes[1, 1].plot(pv, "k")
    axes[1, 1].set_title("вертикальная проекция PV(x)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "lab04_projections.png"), dpi=110)
    plt.close(fig)
    print(f"Графики: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
