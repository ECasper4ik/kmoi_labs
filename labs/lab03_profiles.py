"""Лаба 3. Профили яркости: P(t) = I(p0 + t * (p1 - p0)), билинейная интерполяция."""
import cv2
import numpy as np

import common


def bilinear(g, xs, ys):
    """I(x, y) по четырём соседям с весами из дробных частей координат."""
    h, w = g.shape
    xs, ys = np.clip(xs, 0, w - 1), np.clip(ys, 0, h - 1)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = xs - x0, ys - y0
    g = g.astype(float)
    return (1 - wy) * ((1 - wx) * g[y0, x0] + wx * g[y0, x1]) + wy * ((1 - wx) * g[y1, x0] + wx * g[y1, x1])


def profile(g, p0, p1, width=1):
    """Профиль вдоль p0 -> p1 (точки (x, y)), усреднённый по ширине полосы."""
    common.check(g)
    (x0, y0), (x1, y1) = p0, p1
    length = np.hypot(x1 - x0, y1 - y0)
    if length == 0:
        raise ValueError("Концы отрезка совпадают")
    t = np.linspace(0, 1, int(np.ceil(length)) + 1)  # ~1 отсчёт на пиксель
    xs, ys = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    nx, ny = -(y1 - y0) / length, (x1 - x0) / length  # нормаль к линии
    offs = np.linspace(-(width - 1) / 2, (width - 1) / 2, width)
    return np.mean([bilinear(g, xs + o * nx, ys + o * ny) for o in offs], axis=0), xs, ys


def edges(p, thr):
    """Индексы перепадов: |P[i+1] - P[i]| > thr."""
    return np.flatnonzero(np.abs(np.diff(p)) > thr)


def main():
    args = common.parse(__doc__)
    g = common.load(args.image) if args.image else common.demo_objects()[0]
    h, w = g.shape
    p0, p1 = (5, h // 4), (w - 6, h // 2)
    p, xs, ys = profile(g, p0, p1)
    ref = cv2.remap(g.astype(np.float32), xs.astype(np.float32)[None], ys.astype(np.float32)[None],
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
    print(f"max|diff| с cv2.remap: {np.abs(p - ref).max():.4f}; перепады: {edges(p, 20)}")
    print(f"время: {common.ms(profile, g, p0, p1):.3f} мс")
    noisy = common.gauss(g, 15)
    for w_ in (1, 9):  # усреднение по ширине снижает шум ~ в sqrt(w) раз
        rms = np.sqrt(np.mean((profile(noisy, p0, p1, w_)[0] - profile(g, p0, p1, w_)[0]) ** 2))
        print(f"шум s=15, ширина {w_}: RMS ошибки профиля {rms:.1f}")
    common.plot({"профиль": p, "шум, ширина 1": profile(noisy, p0, p1)[0],
                 "шум, ширина 9": profile(noisy, p0, p1, 9)[0]}, args.out, "lab03.png")


if __name__ == "__main__":
    main()
