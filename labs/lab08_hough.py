"""Лаба 8. Хаф: rho = x cos(theta) + y sin(theta); пики аккумулятора -- прямые."""
import cv2
import numpy as np

import common


def hough_lines(edges, step=1.0):
    """Аккумулятор (rho, theta); каждая точка контура голосует за все свои прямые."""
    common.check(edges)
    ys, xs = np.nonzero(edges)
    diag = int(np.ceil(np.hypot(*edges.shape)))
    thetas = np.deg2rad(np.arange(0, 180, step))
    acc = np.zeros((2 * diag + 1, thetas.size), np.int32)
    for j, th in enumerate(thetas):
        rho = np.rint(xs * np.cos(th) + ys * np.sin(th)).astype(int) + diag
        acc[:, j] = np.bincount(rho, minlength=acc.shape[0])
    return acc, diag, thetas


def peaks(acc, thr, nms=9, top=6):
    """Локальные максимумы выше порога (без подавления немаксимумов -- облако пиков)."""
    a = acc.astype(np.float32)
    p = np.argwhere((a == cv2.dilate(a, np.ones((nms, nms), np.uint8))) & (a >= thr))
    return p[np.argsort(-a[p[:, 0], p[:, 1]])][:top]


def hough_circles(edges, r_min, r_max, min_score=0.6):
    """Окружности: для каждого r точки голосуют за центры (x - r cos t, y - r sin t)."""
    ys, xs = np.nonzero(edges)
    h, w = edges.shape
    found = []
    for r in range(r_min, r_max + 1):
        n = max(36, int(2 * np.pi * r))
        acc = np.zeros(h * w)
        for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
            a, b = np.rint(xs - r * np.cos(t)).astype(int), np.rint(ys - r * np.sin(t)).astype(int)
            ok = (a >= 0) & (a < w) & (b >= 0) & (b < h)
            acc += np.bincount(b[ok] * w + a[ok], minlength=h * w)
        acc = cv2.boxFilter(acc.reshape(h, w).astype(np.float32), -1, (3, 3), normalize=False)
        y, x = np.unravel_index(acc.argmax(), acc.shape)
        if acc.max() / n >= min_score:
            found.append((float(acc.max() / n), int(x), int(y), r))
    return sorted(found, reverse=True)[:1]


def main():
    args = common.parse(__doc__)
    g = common.load(args.image) if args.image else common.demo_lines()
    edges = cv2.Canny(g, 50, 150)
    acc, diag, th = hough_lines(edges)
    pk = peaks(acc, 0.5 * acc.max())
    print("наши (rho, theta°, голоса):", [(int(r - diag), round(float(np.rad2deg(th[t])), 1), int(acc[r, t])) for r, t in pk[:3]])
    ref = cv2.HoughLines(edges, 1, np.pi / 180, int(0.5 * acc.max()))
    print("cv2.HoughLines:", [(round(float(r), 1), round(float(np.rad2deg(t)), 1)) for r, t in ref[:3, 0]])
    print("окружность (score, x, y, r):", hough_circles(edges, 25, 45))
    print("время шаг theta 2/1/0.5°:", [f"{common.ms(hough_lines, edges, s, repeat=1):.1f} мс" for s in (2, 1, 0.5)])
    noisy = cv2.Canny(cv2.GaussianBlur(common.gauss(g, 25), (5, 5), 1.4), 50, 150)
    a2, _, _ = hough_lines(noisy)
    print(f"шум s=25: пиков {len(peaks(a2, 0.5 * a2.max()))} (без шума {len(pk)}), контурных точек {int((noisy > 0).sum())}")
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for r, t in pk:
        a, b, rho = np.cos(th[t]), np.sin(th[t]), r - diag
        cv2.line(vis, (int(a * rho - 999 * b), int(b * rho + 999 * a)), (int(a * rho + 999 * b), int(b * rho - 999 * a)), (0, 0, 255), 1)
    common.grid([(g, "исходное"), (edges, "контуры"), (vis, "найденные прямые"), (noisy, "контуры при шуме")],
                args.out, "lab08.png", 4)


if __name__ == "__main__":
    main()
