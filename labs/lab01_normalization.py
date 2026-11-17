"""Лабораторная работа 1. Яркостная нормализация изображения.

Линейная адаптивная гистограммная обработка (методичка, раздел про LUT):

    LUT[i] = 255 * (i - Imin) / (Imax - Imin),   Imin <= i <= Imax,

где Imin, Imax -- минимальная и максимальная яркость, РЕАЛЬНО присутствующие
на изображении (края носителя гистограммы, а не min/max значений счётчиков).

Запуск:
    python lab01_normalization.py --image path/to/img.png
    python lab01_normalization.py --demo          # синтетическое изображение
    python lab01_normalization.py --selftest      # проверка на матрице 3x3
"""

import argparse
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

LEVELS = 256  # число градаций яркости для uint8


# --------------------------------------------------------------------------
# Ввод и проверки
# --------------------------------------------------------------------------
def load_image(path):
    """Загрузить изображение с проверкой существования файла.

    Args:
        path: путь к файлу изображения.

    Returns:
        BGR-изображение uint8 формы (H, W, 3).

    Raises:
        FileNotFoundError: файла нет.
        ValueError: файл не удалось декодировать (cv2.imread вернул None).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    # cv2.imread не бросает исключение, а молча возвращает None.
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось декодировать изображение: {path}")
    return img


def _check_gray(gray):
    """Проверить, что вход -- непустое 2D-изображение uint8."""
    if not isinstance(gray, np.ndarray) or gray.size == 0:
        raise ValueError("Пустое изображение")
    if gray.ndim != 2:
        raise ValueError(f"Ожидалось 2D (grayscale), получено ndim={gray.ndim}")
    if gray.dtype != np.uint8:
        raise ValueError(f"Ожидался uint8, получен {gray.dtype}")


# --------------------------------------------------------------------------
# Ядро лабораторной: ручная реализация (без cv2.normalize / cv2.LUT)
# --------------------------------------------------------------------------
def build_normalization_lut(i_min, i_max):
    """Построить таблицу LUT линейной нормализации [i_min, i_max] -> [0, 255].

    Прямая f(i) = a * i + b, a = 255 / (i_max - i_min), b = -a * i_min
    получена из условий f(i_min) = 0, f(i_max) = 255.

    Args:
        i_min: минимальная яркость на изображении.
        i_max: максимальная яркость на изображении.

    Returns:
        np.ndarray uint8 длины 256. Вне [i_min, i_max] значения зажаты
        (на изображении таких яркостей нет, но таблица должна быть полной).
    """
    if i_max <= i_min:
        # Вырожденный случай: делить на ноль нельзя. Договорённость --
        # тождественное отображение (изображение однородное, растягивать нечего).
        return np.arange(LEVELS, dtype=np.uint8)
    idx = np.arange(LEVELS, dtype=np.float64)
    # float64: промежуточное (i - i_min) * 255 в uint8 переполнилось бы (mod 256).
    lut = 255.0 * (idx - i_min) / float(i_max - i_min)
    # np.rint -- округление, а не усечение (иначе 32.9 -> 32, а не 33);
    # clip -- защита от выхода за [0, 255] вне рабочего диапазона.
    return np.clip(np.rint(lut), 0, 255).astype(np.uint8)


def normalize_manual(gray, i_min=None, i_max=None):
    """Яркостная нормализация полутонового изображения через LUT.

    Args:
        gray: 2D uint8.
        i_min: нижняя граница; по умолчанию gray.min().
        i_max: верхняя граница; по умолчанию gray.max().

    Returns:
        Нормализованное 2D uint8 того же размера.
    """
    _check_gray(gray)
    i_min = int(gray.min()) if i_min is None else int(i_min)
    i_max = int(gray.max()) if i_max is None else int(i_max)
    lut = build_normalization_lut(i_min, i_max)
    # Индексация массивом: lut[gray] -- это и есть применение таблицы к каждому
    # пикселю за один проход, сложность O(N), N = H*W.
    return lut[gray]


def normalize_naive_loops(gray):
    """Эталонная «в лоб» реализация с циклами по пикселям (для замера времени).

    Считает ту же формулу, но поэлементно, без LUT. Медленная: O(N) в Python.
    """
    _check_gray(gray)
    i_min, i_max = int(gray.min()), int(gray.max())
    out = np.empty_like(gray)
    if i_max == i_min:
        return gray.copy()
    scale = 255.0 / (i_max - i_min)
    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            out[y, x] = int(round((int(gray[y, x]) - i_min) * scale))
    return out


def normalize_robust(gray, low_pct=1.0, high_pct=99.0):
    """Устойчивая нормализация: границы по перцентилям, хвосты усекаются.

    Выброс (один яркий пиксель) не определяет Imax. Цена: значения ниже
    p_low и выше p_high насыщаются в 0 / 255, то есть отображение
    перестаёт быть взаимно однозначным -- различимость в хвостах теряется.
    """
    _check_gray(gray)
    p_lo, p_hi = np.percentile(gray, [low_pct, high_pct])
    lut = build_normalization_lut(int(round(p_lo)), int(round(p_hi)))
    # Для яркостей вне [p_lo, p_hi] нужно насыщение, а в LUT они зажаты clip-ом:
    # ниже p_lo -> 0 (clip снизу), выше p_hi -> 255 (clip сверху). Корректно.
    return lut[gray]


# --------------------------------------------------------------------------
# Цветные изображения
# --------------------------------------------------------------------------
def normalize_color_value(bgr):
    """Нормализация цветного изображения по яркости (канал V в HSV).

    Цветовое пространство HSV выбрано, чтобы отделить яркость (V) от
    цветности (H, S): растягиваем только V, оттенок и насыщенность не трогаем.
    """
    if bgr is None or bgr.size == 0 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Ожидалось непустое 3-канальное изображение")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)  # OpenCV хранит BGR, не RGB
    hsv[..., 2] = normalize_manual(np.ascontiguousarray(hsv[..., 2]))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def normalize_color_per_channel(bgr):
    """Наивная нормализация каждого канала по своим min/max (для сравнения).

    У каналов разные Imin/Imax -> разные коэффициенты растяжения ->
    баланс цветов сдвигается (появляется цветной оттенок).
    """
    out = np.empty_like(bgr)
    for c in range(3):
        out[..., c] = normalize_manual(np.ascontiguousarray(bgr[..., c]))
    return out


# --------------------------------------------------------------------------
# Тестовые данные, шум, замеры
# --------------------------------------------------------------------------
def make_demo_image(h=256, w=256, lo=100, hi=140, seed=0):
    """Синтетическое низкоконтрастное изображение: градиент + круг в [lo, hi]."""
    rng = np.random.default_rng(seed)
    base = np.tile(np.linspace(lo, hi, w), (h, 1))
    yy, xx = np.mgrid[0:h, 0:w]
    base[(yy - h // 2) ** 2 + (xx - w // 2) ** 2 < (h // 4) ** 2] = lo + 0.7 * (hi - lo)
    base += rng.normal(0, 1.0, base.shape)  # слабый гауссов шум
    return np.clip(np.rint(base), lo, hi).astype(np.uint8)


def add_outlier(gray, count=1, value=255, seed=1):
    """Добавить `count` пикселей-выбросов яркости `value` (импульсный шум)."""
    rng = np.random.default_rng(seed)
    out = gray.copy()
    ys = rng.integers(0, gray.shape[0], count)
    xs = rng.integers(0, gray.shape[1], count)
    out[ys, xs] = value
    return out


def timeit(func, *args, repeat=5):
    """Минимальное время выполнения func(*args) из `repeat` запусков, секунды."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        func(*args)
        best = min(best, time.perf_counter() - t0)
    return best


# --------------------------------------------------------------------------
# Визуализация
# --------------------------------------------------------------------------
def _show_pair(axes_col, img, title):
    """Нарисовать изображение (сверху) и его гистограмму (снизу)."""
    ax_img, ax_hist = axes_col
    ax_img.imshow(img, cmap="gray", vmin=0, vmax=255)  # фиксируем шкалу!
    ax_img.set_title(title, fontsize=9)
    ax_img.axis("off")
    ax_hist.hist(img.ravel(), bins=LEVELS, range=(0, LEVELS), color="k")
    ax_hist.set_xlim(0, 255)
    ax_hist.set_yscale("log")


def plot_steps(images, titles, out_path):
    """Сохранить subplot: ряд изображений и под ними гистограммы."""
    fig, axes = plt.subplots(2, len(images), figsize=(3.2 * len(images), 6))
    for col, (img, title) in enumerate(zip(images, titles)):
        _show_pair(axes[:, col], img, title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Самопроверка на матрице 3x3 (то, что просят показать на защите)
# --------------------------------------------------------------------------
def selftest():
    """Проверить формулу на 3x3 вручную и сверить с cv2.normalize."""
    a = np.array([[100, 110, 120],
                  [130, 140, 100],
                  [105, 115, 125]], dtype=np.uint8)
    # a = 255 / (140 - 100) = 6.375; f(120) = 20 * 6.375 = 127.5 -> 128
    expected = np.array([[0, 64, 128],
                         [191, 255, 0],
                         [32, 96, 159]], dtype=np.uint8)
    got = normalize_manual(a)
    assert np.array_equal(got, expected), f"\n{got}\n!=\n{expected}"

    # Выброс: Imax=255, a = 255/155; пиксель 120 -> 33 (случай из разбора)
    b = a.copy()
    b[0, 0] = 255
    b[1, 1] = 100  # Imin остаётся 100
    assert normalize_manual(b)[0, 2] == 33

    # Вырожденный случай: константное изображение не должно давать NaN/ошибку
    c = np.full((3, 3), 120, dtype=np.uint8)
    assert np.array_equal(normalize_manual(c), c)
    print("selftest: OK")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def run(gray_or_bgr, out_dir):
    """Полный прогон: сравнение с cv2, замеры, тест на выброс, графики."""
    os.makedirs(out_dir, exist_ok=True)
    is_color = gray_or_bgr.ndim == 3
    # Яркость для основной части лабы: BGR -> Gray (взвешенная сумма
    # 0.299 R + 0.587 G + 0.114 B, учитывает чувствительность глаза).
    gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY) if is_color else gray_or_bgr

    manual = normalize_manual(gray)

    # 5. Сравнение с эталоном OpenCV (там вычисляется в float и округляется)
    ref = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    diff = np.abs(manual.astype(np.int16) - ref.astype(np.int16))
    print(f"[сравнение с cv2.normalize] max|diff| = {diff.max()}, "
          f"доля отличающихся пикселей = {np.mean(diff > 0):.4%}")

    # 6. Замеры времени (цикл Python только на уменьшенной копии)
    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    t_lut = timeit(normalize_manual, gray)
    t_cv = timeit(cv2.normalize, gray, None, 0, 255, cv2.NORM_MINMAX)
    t_loop = timeit(normalize_naive_loops, small, repeat=1)
    t_lut_small = timeit(normalize_manual, small)
    print(f"[время] LUT numpy {gray.shape}: {t_lut * 1e3:.3f} мс | "
          f"cv2.normalize: {t_cv * 1e3:.3f} мс")
    print(f"[время] {small.shape}: циклы Python {t_loop * 1e3:.1f} мс | "
          f"LUT {t_lut_small * 1e3:.3f} мс")

    # 7. Тест на шум/артефакты: один пиксель-выброс 255
    noisy = add_outlier(gray, count=1, value=255)
    plot_steps(
        [gray, manual, noisy, normalize_manual(noisy), normalize_robust(noisy)],
        ["исходное", "нормализация", "+1 выброс 255",
         "нормализация (выброс)", "перцентили 1..99"],
        os.path.join(out_dir, "lab01_gray.png"),
    )
    for name, img in [("без выброса", normalize_manual(gray)),
                      ("с выбросом", normalize_manual(noisy)),
                      ("с выбросом, перцентили", normalize_robust(noisy))]:
        print(f"[контраст] {name}: std = {img.std():.1f}")

    if is_color:
        plot_color = [gray_or_bgr, normalize_color_per_channel(gray_or_bgr),
                      normalize_color_value(gray_or_bgr)]
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.5))
        for a, im, t in zip(ax, plot_color,
                            ["исходное", "по каналам (сдвиг цвета)", "только V в HSV"]):
            a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))  # matplotlib ждёт RGB
            a.set_title(t, fontsize=9)
            a.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "lab01_color.png"), dpi=120)
        plt.close(fig)
    print(f"Графики сохранены в {out_dir}")


def main():
    """Точка входа командной строки."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--image", help="путь к изображению")
    parser.add_argument("--demo", action="store_true", help="синтетический пример")
    parser.add_argument("--selftest", action="store_true", help="проверка 3x3")
    parser.add_argument("--out", default="out_lab01", help="папка для графиков")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return 0
    if args.demo:
        run(make_demo_image(), args.out)
        return 0
    if not args.image:
        parser.error("нужен --image, --demo или --selftest")
    run(load_image(args.image), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
