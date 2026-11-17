"""Общие утилиты: ввод, шум, метрики, тестовые изображения, графики."""
import argparse
import os
import time

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path, color=False):
    """Загрузить изображение (ошибка, если файла нет или он не читается)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось прочитать: {path}")
    return img


def check(img):
    """Проверить, что вход -- непустое 2D-изображение uint8."""
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.size == 0 or img.dtype != np.uint8:
        raise ValueError("Нужно непустое 2D-изображение uint8")


def parse(doc, extra=()):
    """Аргументы: --image (иначе демо), --out; extra -- список (имя, kwargs)."""
    p = argparse.ArgumentParser(description=doc)
    p.add_argument("--image", help="путь к изображению (иначе демо-данные)")
    p.add_argument("--out", default="out", help="папка для графиков")
    for name, kw in extra:
        p.add_argument(name, **kw)
    return p.parse_args()


def ms(func, *args, repeat=3):
    """Лучшее время func(*args) в миллисекундах."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        func(*args)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def psnr(a, b):
    """PSNR (дБ) между uint8-изображениями."""
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def iou(a, b):
    """Intersection over Union булевых масок."""
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else np.logical_and(a, b).sum() / union


def f1(ref, test, tol=1):
    """F1 карт контуров с допуском tol пикселей."""
    k = np.ones((2 * tol + 1,) * 2, np.uint8)
    r, t = ref > 0, test > 0
    p = (t & (cv2.dilate(r.astype(np.uint8), k) > 0)).sum() / max(t.sum(), 1)
    q = (r & (cv2.dilate(t.astype(np.uint8), k) > 0)).sum() / max(r.sum(), 1)
    return 0.0 if p + q == 0 else 2 * p * q / (p + q)


def gauss(img, sigma, seed=0):
    """Гауссов шум N(0, sigma^2) с насыщением."""
    noisy = img + np.random.default_rng(seed).normal(0, sigma, img.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def salt_pepper(img, amount, seed=0):
    """Импульсный шум: доля amount пикселей -> 0 или 255."""
    r = np.random.default_rng(seed).random(img.shape)
    out = img.copy()
    out[r < amount / 2] = 0
    out[(r >= amount / 2) & (r < amount)] = 255
    return out


def windows(img, k):
    """Окна k x k вокруг каждого пикселя, форма (H, W, k, k); края -- reflect."""
    return np.lib.stride_tricks.sliding_window_view(np.pad(img, k // 2, mode="reflect"), (k, k))


def demo_objects(h=240, w=320, noise=8.0, seed=0):
    """Светлые объекты на фоне с градиентом освещения -> (изображение, маска)."""
    yy, xx = np.mgrid[0:h, 0:w]
    bg = 40.0 + 70.0 * xx / w + 20.0 * yy / h
    mask = np.zeros((h, w), np.uint8)
    for cx, cy, r in [(.15, .25, .054), (.40, .30, .042), (.70, .22, .060),
                      (.25, .70, .048), (.55, .72, .066), (.85, .65, .036)]:
        cv2.circle(mask, (int(cx * w), int(cy * h)), int(r * h), 1, -1)
    cv2.rectangle(mask, (int(.45 * w), int(.50 * h)), (int(.50 * w), int(.58 * h)), 1, -1)
    img = np.where(mask > 0, bg + 90, bg) + np.random.default_rng(seed).normal(0, noise, (h, w))
    return np.clip(np.rint(img), 0, 255).astype(np.uint8), mask > 0


def demo_lines(h=240, w=320):
    """Белые прямые и окружность r=35 на чёрном фоне."""
    img = np.zeros((h, w), np.uint8)
    cv2.line(img, (10, 40), (300, 40), 255, 2)
    cv2.line(img, (60, 10), (60, 230), 255, 2)
    cv2.line(img, (100, 220), (300, 100), 255, 2)
    cv2.circle(img, (220, 170), 35, 255, 2)
    return img


def demo_text(angle=0.0):
    """Три строки тёмного текста на светлом фоне, повёрнутые на angle градусов."""
    img = np.full((200, 420), 235, np.uint8)
    for i, s in enumerate(["KMOI LAB PROJECTIONS", "HORIZONTAL AND VERTICAL", "TEXT LINES 12345"]):
        cv2.putText(img, s, (20, 50 + 55 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 20, 2, cv2.LINE_AA)
    m = cv2.getRotationMatrix2D((210, 100), angle, 1.0)
    return gauss(cv2.warpAffine(img, m, (420, 200), borderValue=235), 3.0)


def demo_regions(h=200, w=300, noise=10.0, seed=0):
    """Три яркостных класса (40/120/200) + шум -> (изображение, метки 0..2)."""
    lab = np.zeros((h, w), np.int32)
    cv2.rectangle(lab, (30, 30), (140, 170), 1, -1)
    cv2.circle(lab, (215, 100), 55, 2, -1)
    img = np.array([40.0, 120.0, 200.0])[lab] + np.random.default_rng(seed).normal(0, noise, lab.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8), lab


def grid(items, out, name, ncols=3):
    """Сохранить subplot из пар (изображение, заголовок); 3 канала = BGR."""
    nrows = -(-len(items) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (img, title) in zip(axes.ravel(), items):
        img = img.astype(np.uint8) * 255 if img.dtype == bool else img
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img,
                  cmap="gray", vmin=0, vmax=255 if img.dtype == np.uint8 else None)
        ax.set_title(title, fontsize=9)
    save(fig, out, name)


def plot(curves, out, name):
    """Сохранить график кривых {подпись: массив}."""
    fig, ax = plt.subplots(figsize=(7, 3))
    for label, y in curves.items():
        ax.plot(y, label=label)
    ax.legend()
    save(fig, out, name)


def save(fig, out, name):
    """Сохранить фигуру в out/name и закрыть."""
    os.makedirs(out, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=100)
    plt.close(fig)
