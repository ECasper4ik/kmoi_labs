"""Общие утилиты для лабораторных работ по КМОИ.

Здесь нет алгоритмов, которые нужно защищать: только ввод/вывод,
синтетические тестовые изображения, шум, метрики и сохранение графиков.
"""

import argparse
import os
import time

import cv2
import matplotlib

matplotlib.use("Agg")  # без GUI: графики сохраняются в файлы
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# --------------------------------------------------------------------------
# Ввод / вывод
# --------------------------------------------------------------------------
def load_image(path, color=False):
    """Загрузить изображение с проверкой существования файла.

    Args:
        path: путь к файлу.
        color: True -- BGR uint8 (H, W, 3); False -- Gray uint8 (H, W).

    Raises:
        FileNotFoundError: файла нет.
        ValueError: файл не декодируется (cv2.imread возвращает None).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(path, flag)
    if img is None:
        raise ValueError(f"Не удалось декодировать изображение: {path}")
    return img


def to_gray(img):
    """BGR -> Gray (0.299 R + 0.587 G + 0.114 B); 2D-вход возвращается как есть."""
    if img is None or img.size == 0:
        raise ValueError("Пустое изображение")
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Неподдерживаемая форма изображения: {img.shape}")


def check_gray_u8(img):
    """Проверить, что вход -- непустое 2D-изображение uint8."""
    if not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("Пустое изображение")
    if img.ndim != 2:
        raise ValueError(f"Ожидалось 2D (grayscale), получено ndim={img.ndim}")
    if img.dtype != np.uint8:
        raise ValueError(f"Ожидался uint8, получен {img.dtype}")


def build_parser(description):
    """Парсер аргументов, общий для всех лаб: --image / --demo / --out."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--image", help="путь к изображению")
    parser.add_argument("--demo", action="store_true",
                        help="синтетический пример (файл не нужен)")
    parser.add_argument("--out", default="out", help="папка для графиков")
    return parser


def get_input(args, demo_factory, color=False):
    """Взять изображение из --image либо из demo_factory()."""
    if args.image:
        return load_image(args.image, color=color)
    return demo_factory()


def save_grid(items, out_path, ncols=None, size=3.2):
    """Сохранить subplot из списка (изображение, заголовок).

    2D-массивы рисуются в градациях серого с фиксированной шкалой 0..255
    (uint8) -- иначе matplotlib сам растянет контраст и скроет эффект.
    3-канальные считаются BGR и переводятся в RGB для matplotlib.
    """
    n = len(items)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (img, title) in zip(axes, items):
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        if img.ndim == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        elif img.dtype == np.uint8:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path) or ".")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def ensure_dir(path):
    """Создать папку, если её нет."""
    os.makedirs(path, exist_ok=True)


# --------------------------------------------------------------------------
# Метрики, шум, время
# --------------------------------------------------------------------------
def timeit(func, *args, repeat=5, **kwargs):
    """Минимальное время func(*args) из `repeat` запусков, секунды."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best


def psnr(ref, test):
    """PSNR в дБ между двумя uint8-изображениями (inf, если они равны)."""
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def iou(a, b):
    """Intersection over Union двух булевых масок."""
    a, b = a.astype(bool), b.astype(bool)
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else np.logical_and(a, b).sum() / union


def f1_score(ref, test, tol=1):
    """F1 карт контуров с допуском в `tol` пикселей (контур редко совпадает точно)."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * tol + 1, 2 * tol + 1))
    ref_b, test_b = ref > 0, test > 0
    ref_d = cv2.dilate(ref_b.astype(np.uint8), k) > 0
    test_d = cv2.dilate(test_b.astype(np.uint8), k) > 0
    precision = (test_b & ref_d).sum() / max(test_b.sum(), 1)
    recall = (ref_b & test_d).sum() / max(ref_b.sum(), 1)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def add_gaussian_noise(img, sigma, seed=0):
    """Аддитивный гауссов шум N(0, sigma^2) с насыщением до [0, 255]."""
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float64) + rng.normal(0.0, sigma, img.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def add_salt_pepper(img, amount, seed=0):
    """Импульсный шум: доля `amount` пикселей -> 0 или 255 (поровну)."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    r = rng.random(img.shape)
    out[r < amount / 2] = 0
    out[(r >= amount / 2) & (r < amount)] = 255
    return out


# --------------------------------------------------------------------------
# Синтетические тестовые изображения (с эталонной разметкой)
# --------------------------------------------------------------------------
def demo_objects(h=240, w=320, noise=8.0, seed=0):
    """Яркие объекты на фоне с неравномерной освещённостью.

    Returns:
        (img uint8, mask bool) -- изображение и эталонная маска объектов.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    background = 40.0 + 70.0 * (xx / w) + 20.0 * (yy / h)  # градиент освещения
    mask = np.zeros((h, w), np.uint8)
    # Фиксированные объекты (относительные координаты) -- результат воспроизводим.
    for cx, cy, r in [(0.15, 0.25, 0.054), (0.40, 0.30, 0.042), (0.70, 0.22, 0.060),
                      (0.25, 0.70, 0.048), (0.55, 0.72, 0.066), (0.85, 0.65, 0.036)]:
        cv2.circle(mask, (int(cx * w), int(cy * h)), int(r * h), 1, -1)
    cv2.rectangle(mask, (int(0.45 * w), int(0.50 * h)),
                  (int(0.50 * w), int(0.58 * h)), 1, -1)
    img = np.where(mask > 0, background + 90.0, background)
    img += np.random.default_rng(seed).normal(0.0, noise, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8), mask > 0


def demo_lines(h=240, w=320, noise=0.0, seed=0):
    """Тёмный фон, белые линии разных углов и окружность (для Хафа)."""
    img = np.zeros((h, w), np.uint8)
    cv2.line(img, (10, 40), (300, 40), 255, 2)       # горизонталь
    cv2.line(img, (60, 10), (60, 230), 255, 2)       # вертикаль
    cv2.line(img, (100, 220), (300, 100), 255, 2)    # наклонная
    cv2.circle(img, (220, 170), 35, 255, 2)          # окружность r=35
    return add_gaussian_noise(img, noise, seed) if noise else img


def demo_text(h=200, w=420, angle=0.0, seed=0):
    """Тёмный текст на светлом фоне (3 строки), опционально повёрнутый."""
    img = np.full((h, w), 235, np.uint8)
    for i, s in enumerate(["KMOI LAB PROJECTIONS", "HORIZONTAL AND VERTICAL", "TEXT LINES 12345"]):
        cv2.putText(img, s, (20, 50 + 55 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 20, 2,
                    cv2.LINE_AA)
    if angle:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderValue=235)
    return add_gaussian_noise(img, 3.0, seed)


def demo_regions(h=200, w=300, noise=10.0, seed=0):
    """Три яркостных класса (40 / 120 / 200) + шум.

    Returns:
        (img uint8, labels int) -- метки 0, 1, 2 соответствуют уровням.
    """
    labels = np.zeros((h, w), np.int32)
    cv2.rectangle(labels, (30, 30), (140, 170), 1, -1)
    cv2.circle(labels, (215, 100), 55, 2, -1)
    levels = np.array([40, 120, 200], np.float64)
    img = levels[labels] + np.random.default_rng(seed).normal(0, noise, labels.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8), labels


def sliding(img, kh, kw=None, mode="reflect"):
    """Окна kh x kw вокруг каждого пикселя: массив (H, W, kh, kw) (представление).

    Края дополняются зеркально (reflect = BORDER_REFLECT_101 в OpenCV).
    """
    kw = kh if kw is None else kw
    padded = np.pad(img, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode=mode)
    return np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
