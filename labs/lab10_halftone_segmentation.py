"""Лабораторная работа 10. Сегментация полутоновых изображений.

Яркостная сегментация по гистограмме (методичка): число мод гистограммы
= число яркостных классов, пороги -- во впадинах между модами. Дополнительно:
k-means по яркости (центры классов оптимизируются итеративно) и наращивание
областей от затравки.

Запуск: python lab10_halftone_segmentation.py --demo | --image файл.png
"""

import sys
from collections import deque

import cv2
import numpy as np

import common


def smooth_hist(hist, sigma=3.0):
    """Сглаживание гистограммы гауссовым ядром (иначе шум даёт ложные моды)."""
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    padded = np.pad(hist.astype(np.float64), radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def find_modes(hist_s, min_height=0.05, min_dist=20):
    """Моды (локальные максимумы) сглаженной гистограммы.

    Args:
        min_height: минимальная высота пика относительно максимума.
        min_dist: минимальное расстояние между модами (более низкая из близких отбрасывается).
    """
    peaks = [i for i in range(1, len(hist_s) - 1)
             if hist_s[i] > hist_s[i - 1] and hist_s[i] >= hist_s[i + 1]
             and hist_s[i] >= min_height * hist_s.max()]
    peaks.sort(key=lambda i: -hist_s[i])
    chosen = []
    for p in peaks:
        if all(abs(p - c) >= min_dist for c in chosen):
            chosen.append(p)
    return sorted(chosen)


def valley_thresholds(gray, sigma=3.0, min_height=0.05, min_dist=20):
    """Пороги = минимумы гистограммы между соседними модами.

    Returns:
        (thresholds, modes). Для унимодальной гистограммы порогов нет.
    """
    common.check_gray_u8(gray)
    hist = np.bincount(gray.ravel(), minlength=256)
    hs = smooth_hist(hist, sigma)
    modes = find_modes(hs, min_height, min_dist)
    thresholds = [m1 + int(np.argmin(hs[m1:m2 + 1])) for m1, m2 in zip(modes[:-1], modes[1:])]
    return thresholds, modes


def segment_by_thresholds(gray, thresholds):
    """Метки классов 0..len(thresholds): digitize по порогам."""
    return np.digitize(gray, thresholds).astype(np.int32)


def kmeans_1d(gray, k, n_iter=100):
    """k-means по яркости (алгоритм Ллойда) на гистограмме: O(256 k) на итерацию.

    Шаги: каждый уровень -> ближайший центр; центр -> взвешенное среднее
    своих уровней. Сходится к локальному минимуму суммы квадратов расстояний.

    Returns:
        (centers отсортированные, labels 0..k-1 в порядке возрастания яркости)
    """
    common.check_gray_u8(gray)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    levels = np.arange(256, dtype=np.float64)
    nz = levels[hist > 0]
    if nz.size < k:
        raise ValueError("Уровней яркости меньше, чем кластеров")
    # Начальные центры: метод наиболее удалённых точек среди значимых уровней
    # (>= 0.1% пикселей, чтобы редкий выброс не стал центром). Квантильная
    # инициализация плоха, когда один класс занимает большую часть кадра.
    candidates = levels[hist >= 0.001 * hist.sum()]
    if candidates.size < k:
        candidates = nz
    centers = [levels[int(np.argmax(hist))]]
    while len(centers) < k:
        dist = np.min(np.abs(candidates[:, None] - np.array(centers)[None, :]), axis=1)
        centers.append(candidates[int(np.argmax(dist))])
    centers = np.array(centers, dtype=np.float64)
    for _ in range(n_iter):
        assign = np.argmin(np.abs(levels[:, None] - centers[None, :]), axis=1)
        new = centers.copy()
        for j in range(k):
            w = hist * (assign == j)
            if w.sum() > 0:
                new[j] = (levels * w).sum() / w.sum()
        if np.allclose(new, centers, atol=1e-3):
            centers = new
            break
        centers = new
    order = np.argsort(centers)
    centers = centers[order]
    lut = np.argmin(np.abs(levels[:, None] - centers[None, :]), axis=1).astype(np.int32)
    return centers, lut[gray]


def region_growing(gray, seed, tol):
    """Наращивание области от затравки seed = (x, y).

    Пиксель присоединяется, если |I - среднее области| <= tol; соседство 8-связное.
    Среднее обновляется по ходу (сумма и число пикселей).
    """
    common.check_gray_u8(gray)
    h, w = gray.shape
    x0, y0 = seed
    if not (0 <= x0 < w and 0 <= y0 < h):
        raise ValueError("Затравка вне изображения")
    mask = np.zeros((h, w), bool)
    mask[y0, x0] = True
    total, count = float(gray[y0, x0]), 1
    queue = deque([(x0, y0)])
    while queue:
        x, y = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not mask[ny, nx]:
                    if abs(float(gray[ny, nx]) - total / count) <= tol:
                        mask[ny, nx] = True
                        total += float(gray[ny, nx])
                        count += 1
                        queue.append((nx, ny))
    return mask


def main():
    """Точка входа."""
    args = common.build_parser(__doc__).parse_args()
    if not (args.demo or args.image):
        print("Нужен --demo или --image")
        return 1
    if args.image:
        gray, truth = common.to_gray(common.load_image(args.image)), None
    else:
        gray, truth = common.demo_regions(noise=10.0)
    common.ensure_dir(args.out)

    thr, modes = valley_thresholds(gray)
    kind = {1: "унимодальная", 2: "бимодальная"}.get(len(modes), "мультимодальная")
    print(f"[гистограмма] {kind}, моды {modes}, пороги {thr}")
    seg_hist = segment_by_thresholds(gray, thr)
    centers, seg_km = kmeans_1d(gray, max(len(modes), 2))
    print(f"[k-means 1D] центры {np.round(centers, 1)}")

    samples = gray.reshape(-1, 1).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    _, _, ref_centers = cv2.kmeans(samples, max(len(modes), 2), None, crit, 5,
                                   cv2.KMEANS_PP_CENTERS)
    print(f"[cv2.kmeans] центры {np.round(np.sort(ref_centers.ravel()), 1)}")

    if truth is not None:
        acc = lambda seg: float(np.mean(seg == truth))  # noqa: E731
        print(f"[точность пикселей] пороги по гистограмме {acc(seg_hist):.3f} | "
              f"k-means {acc(seg_km):.3f}")
        grown = region_growing(gray, (215, 100), tol=30)
        print(f"[наращивание областей] IoU круга = {common.iou(grown, truth == 2):.3f}")
    else:
        grown = region_growing(gray, (gray.shape[1] // 2, gray.shape[0] // 2), tol=20)

    print(f"[время] пороги по гистограмме {common.timeit(valley_thresholds, gray) * 1e3:.2f} мс | "
          f"k-means 1D {common.timeit(kmeans_1d, gray, 3) * 1e3:.2f} мс | "
          f"наращивание {common.timeit(region_growing, gray, (215, 100), 30, repeat=1) * 1e3:.0f} мс")

    # Тест на шум: сильный шум без/с предфильтрацией медианой.
    noisy = common.add_gaussian_noise(gray, 30.0)
    filt = cv2.medianBlur(noisy, 7)
    t_n, m_n = valley_thresholds(noisy)
    t_f, m_f = valley_thresholds(filt)
    print(f"[шум s=30] моды без фильтра {m_n}, после медианы {m_f}")
    if truth is not None:
        print(f"[шум s=30] точность: без фильтра {acc(segment_by_thresholds(noisy, t_n)):.3f} "
              f"| после медианы {acc(segment_by_thresholds(filt, t_f)):.3f}")

    scale = 255 // max(int(seg_hist.max()), 1)
    common.save_grid([(gray, "исходное"), ((seg_hist * scale).astype(np.uint8), "по гистограмме"),
                      ((seg_km * scale).astype(np.uint8), "k-means"), (grown, "наращивание областей"),
                      (noisy, "шум s=30"),
                      ((segment_by_thresholds(filt, t_f) * scale).astype(np.uint8), "шум + медиана")],
                     f"{args.out}/lab10_segmentation.png", ncols=3)
    print(f"Графики: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
