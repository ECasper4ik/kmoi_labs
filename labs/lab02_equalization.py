"""Лабораторная работа 2. Эквализация гистограммы.

Методичка: LUT[i] = 255 * SUM(Hist, Imin, i) / SUM(Hist, Imin, Imax),
то есть яркость заменяется нормированной кумулятивной гистограммой (CDF).
Цель -- «выровнять» число пикселей на разных уровнях яркости.

Запуск: python lab02_equalization.py --demo | --image файл.png
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

import common
from lab01_normalization import normalize_manual

LEVELS = 256


def histogram(gray):
    """Гистограмма яркости: hist[i] = число пикселей со значением i."""
    common.check_gray_u8(gray)
    return np.bincount(gray.ravel(), minlength=LEVELS).astype(np.int64)


def equalization_lut(hist, variant="methodichka"):
    """Построить LUT эквализации по гистограмме.

    Args:
        hist: массив длины 256.
        variant: "methodichka" -- формула методички, CDF / N;
            "opencv" -- (CDF - CDF_min) / (N - CDF_min): самый тёмный
            пиксель уходит ровно в 0, как в cv2.equalizeHist.

    Returns:
        LUT uint8 длины 256.
    """
    hist = np.asarray(hist, dtype=np.float64)
    total = hist.sum()
    if total == 0:
        raise ValueError("Пустая гистограмма")
    nonzero = np.flatnonzero(hist)
    i_min, i_max = nonzero[0], nonzero[-1]
    if i_min == i_max:  # однородное изображение: растягивать нечего
        return np.arange(LEVELS, dtype=np.uint8)
    # cdf[i] = SUM(hist, 0, i); ниже Imin гистограмма нулевая => то же, что SUM от Imin.
    cdf = np.cumsum(hist)
    if variant == "methodichka":
        lut = 255.0 * cdf / total
    elif variant == "opencv":
        cdf_min = cdf[i_min]
        lut = 255.0 * (cdf - cdf_min) / (total - cdf_min)
    else:
        raise ValueError(f"Неизвестный variant: {variant}")
    # rint -- округление к ближайшему (как cvRound), clip -- защита от выхода за [0, 255].
    return np.clip(np.rint(lut), 0, 255).astype(np.uint8)


def equalize(gray, variant="methodichka"):
    """Эквализация полутонового изображения через LUT (один проход, O(N))."""
    return equalization_lut(histogram(gray), variant)[gray]


def equalize_color(bgr):
    """Эквализация цветного изображения по яркости V (HSV), цвет не искажается."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = equalize(np.ascontiguousarray(hsv[..., 2]))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def make_demo():
    """Низкоконтрастное изображение: яркости сжаты в узкий диапазон [70, 130]."""
    img, _ = common.demo_objects()
    return np.clip(img.astype(np.float64) * 0.35 + 60, 0, 255).astype(np.uint8)


def plot_hist_cdf(images, titles, path):
    """Строка 1 -- изображения, строка 2 -- гистограммы, строка 3 -- CDF."""
    fig, axes = plt.subplots(3, len(images), figsize=(3.2 * len(images), 8))
    for col, (img, title) in enumerate(zip(images, titles)):
        axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(title, fontsize=9)
        axes[0, col].axis("off")
        hist = histogram(img)
        axes[1, col].bar(np.arange(LEVELS), hist, width=1.0, color="k")
        axes[1, col].set_xlim(0, 255)
        cdf = np.cumsum(hist) / hist.sum()
        axes[2, col].plot(cdf, "k")
        axes[2, col].set_xlim(0, 255)
        axes[2, col].set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    """Точка входа: сравнение, замеры, тест на шум."""
    args = common.build_parser(__doc__).parse_args()
    if not (args.demo or args.image):
        print("Нужен --demo или --image")
        return 1
    img = common.get_input(args, make_demo)
    gray = common.to_gray(img)
    common.ensure_dir(args.out)

    mine = equalize(gray, "methodichka")
    mine_cv = equalize(gray, "opencv")
    ref = cv2.equalizeHist(gray)
    d = np.abs(mine_cv.astype(int) - ref.astype(int))
    print(f"[сравнение] opencv-вариант vs cv2.equalizeHist: max|diff|={d.max()}")
    d2 = np.abs(mine.astype(int) - ref.astype(int))
    print(f"[сравнение] вариант методички vs cv2.equalizeHist: max|diff|={d2.max()} "
          "(отличие только в нормировке: темнейший пиксель не обязательно 0)")

    norm = normalize_manual(gray)
    print(f"[std] исходное {gray.std():.1f} | нормализация {norm.std():.1f} "
          f"| эквализация {mine.std():.1f}")

    print(f"[время] LUT-эквализация {common.timeit(equalize, gray) * 1e3:.3f} мс | "
          f"cv2.equalizeHist {common.timeit(cv2.equalizeHist, gray) * 1e3:.3f} мс")

    # Тест на шум: эквализация усиливает шум в тех областях, где гистограмма плотная.
    noisy = common.add_gaussian_noise(gray, 6.0)
    eq_noisy = equalize(noisy)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(noisy)
    print(f"[шум] std(eq(noisy) - eq(clean)) = "
          f"{(eq_noisy.astype(float) - mine.astype(float)).std():.1f} "
          "(усиление шума эквализацией)")

    plot_hist_cdf([gray, norm, mine, eq_noisy, clahe],
                  ["исходное", "нормализация", "эквализация", "эквализация + шум",
                   "CLAHE + шум"], os.path.join(args.out, "lab02_hist_cdf.png"))
    print(f"Графики: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
