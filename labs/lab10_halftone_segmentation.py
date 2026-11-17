"""Лаба 10. Сегментация полутоновых изображений: пороги во впадинах гистограммы, k-means, наращивание областей."""
from collections import deque

import cv2
import numpy as np

import common


def valley_thresholds(gray, sigma=3, min_height=0.05, min_dist=20):
    """Моды сглаженной гистограммы и пороги -- минимумы между соседними модами."""
    common.check(gray)
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    x = np.arange(-3 * sigma, 3 * sigma + 1)
    ker = np.exp(-x ** 2 / (2 * sigma ** 2))
    hs = np.convolve(np.pad(hist, 3 * sigma, mode="reflect"), ker / ker.sum(), mode="valid")
    cand = [i for i in range(1, 255) if hs[i] > hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] >= min_height * hs.max()]
    modes = []
    for i in sorted(cand, key=lambda i: -hs[i]):  # высокие моды вытесняют близкие низкие
        if all(abs(i - m) >= min_dist for m in modes):
            modes.append(i)
    modes.sort()
    return [a + int(np.argmin(hs[a:b + 1])) for a, b in zip(modes, modes[1:])], modes


def kmeans(gray, k, iters=100):
    """k-means по яркости (Ллойд на гистограмме) -> (центры по возрастанию, метки)."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    lv = np.arange(256.0)
    cand = lv[hist >= 0.001 * hist.sum()]
    c = [float(hist.argmax())]
    while len(c) < k:  # инициализация: самые удалённые значимые уровни
        c.append(cand[np.argmax(np.abs(cand[:, None] - c).min(axis=1))])
    c = np.array(c)
    for _ in range(iters):
        a = np.abs(lv[:, None] - c).argmin(axis=1)
        new = np.array([(lv * hist * (a == j)).sum() / max((hist * (a == j)).sum(), 1e-9) if (a == j).any() else c[j]
                        for j in range(k)])
        if np.allclose(new, c, atol=1e-3):
            break
        c = new
    c.sort()
    return c, np.abs(lv[:, None] - c).argmin(axis=1)[gray]


def region_growing(gray, seed, tol):
    """От затравки (x, y): присоединять 8-соседей с |I - среднее области| <= tol."""
    h, w = gray.shape
    mask = np.zeros((h, w), bool)
    mask[seed[1], seed[0]] = True
    total, cnt, q = float(gray[seed[1], seed[0]]), 1, deque([seed])
    while q:
        x, y = q.popleft()
        for ny in range(max(y - 1, 0), min(y + 2, h)):
            for nx in range(max(x - 1, 0), min(x + 2, w)):
                if not mask[ny, nx] and abs(float(gray[ny, nx]) - total / cnt) <= tol:
                    mask[ny, nx] = True
                    total, cnt = total + float(gray[ny, nx]), cnt + 1
                    q.append((nx, ny))
    return mask


def main():
    args = common.parse(__doc__)
    g, truth = (common.load(args.image), None) if args.image else common.demo_regions()
    thr, modes = valley_thresholds(g)
    seg = np.digitize(g, thr)
    c, km = kmeans(g, max(len(modes), 2))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    ref = np.sort(cv2.kmeans(g.reshape(-1, 1).astype(np.float32), max(len(modes), 2), None, crit, 5, cv2.KMEANS_PP_CENTERS)[2].ravel())
    print(f"моды {modes}, пороги {thr}; центры k-means {np.round(c, 1)}, cv2 {np.round(ref, 1)}")
    seed = (215, 100) if truth is not None else (g.shape[1] // 2, g.shape[0] // 2)
    grown = region_growing(g, seed, 30)
    print(f"время: пороги {common.ms(valley_thresholds, g):.2f} мс, k-means {common.ms(kmeans, g, 3):.2f} мс, "
          f"наращивание {common.ms(region_growing, g, seed, 30, repeat=1):.0f} мс")
    noisy = common.gauss(g, 30)
    filt = cv2.medianBlur(noisy, 7)
    t_n, m_n = valley_thresholds(noisy)
    t_f, m_f = valley_thresholds(filt)
    print(f"шум s=30: моды без фильтра {m_n}, после медианы {m_f}")
    if truth is not None:
        acc = lambda s: float(np.mean(s == truth))  # noqa: E731
        print(f"точность: пороги {acc(seg):.3f}, k-means {acc(km):.3f}, наращивание IoU {common.iou(grown, truth == 2):.3f}; "
              f"шум {acc(np.digitize(noisy, t_n)):.3f} -> после медианы {acc(np.digitize(filt, t_f)):.3f}")
    s = 255 // max(int(seg.max()), 1)
    common.grid([(g, "исходное"), ((seg * s).astype(np.uint8), "по гистограмме"), ((km * s).astype(np.uint8), "k-means"),
                 (grown, "наращивание"), (noisy, "шум s=30"), ((np.digitize(filt, t_f) * s).astype(np.uint8), "шум + медиана")],
                args.out, "lab10.png")


if __name__ == "__main__":
    main()
