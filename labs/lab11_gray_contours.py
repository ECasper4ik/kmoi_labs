"""Лаба 11. Контуры на полутоновых изображениях: градиент |G| = sqrt(gx^2 + gy^2), схема Канни."""
import cv2
import numpy as np

import common
from lab07_linear_filtering import LAPLACE, SOBEL_X, correlate, gauss1d, u8

PREWITT_X = np.array([[-1, 0, 1]] * 3, float)


def gradient(g, kind="sobel"):
    """(gx, gy) оператором Робертса (2x2), Прюитта или Собеля (3x3)."""
    if kind == "roberts":
        p = np.pad(g.astype(float), ((0, 1), (0, 1)), mode="edge")
        return p[1:, 1:] - p[:-1, :-1], p[:-1, 1:] - p[1:, :-1]
    kx = {"sobel": SOBEL_X, "prewitt": PREWITT_X}[kind]
    return correlate(g, kx), correlate(g, kx.T)


def nms(mag, gx, gy):
    """Подавление немаксимумов: пиксель остаётся, если он максимум вдоль градиента (4 направления)."""
    h, w = mag.shape
    q = ((np.rad2deg(np.arctan2(gy, gx)) % 180 + 22.5) // 45).astype(int) % 4
    m = np.pad(mag, 1)
    shifts = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}  # (dy, dx) соседа вдоль градиента
    keep = np.zeros((h, w), bool)
    for d, (dy, dx) in shifts.items():
        a, b = m[1 + dy:1 + dy + h, 1 + dx:1 + dx + w], m[1 - dy:1 - dy + h, 1 - dx:1 - dx + w]
        keep |= (q == d) & (mag >= a) & (mag >= b)
    return np.where(keep, mag, 0)


def canny(g, sigma=1.4, low=0.1, high=0.25):
    """Гаусс -> Собель -> NMS -> гистерезис (пороги -- доли максимума)."""
    k = gauss1d(2 * int(np.ceil(3 * sigma)) + 1, sigma)
    gx, gy = gradient(u8(correlate(correlate(g, k[None]), k[:, None])))
    n = nms(np.hypot(gx, gy), gx, gy)
    _, comp = cv2.connectedComponents((n >= low * n.max()).astype(np.uint8), connectivity=8)
    ids = np.unique(comp[n >= high * n.max()])  # слабые пиксели -- только связанные с сильными
    return (np.isin(comp, ids[ids > 0])).astype(np.uint8) * 255


def main():
    args = common.parse(__doc__)
    g, truth = (common.load(args.image), None) if args.image else common.demo_objects(noise=6)
    gx, gy = gradient(g)
    print("max|gx - cv2.Sobel|:", np.abs(gx - cv2.Sobel(g, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REFLECT_101)).max())
    mine, ref = canny(g), cv2.Canny(g, 50, 150)
    print(f"Канни: наш {int((mine > 0).sum())} пикселей, cv2 {int((ref > 0).sum())}, взаимное F1 {common.f1(ref, mine):.3f}")
    print("время:", {k: f"{common.ms(gradient, g, k):.2f} мс" for k in ("roberts", "sobel")},
          f"canny {common.ms(canny, g, repeat=1):.1f} мс, cv2 {common.ms(cv2.Canny, g, 50, 150):.3f} мс")
    noisy = common.gauss(g, 20)
    if truth is not None:
        t = truth.astype(np.uint8)
        edge = (t - cv2.erode(t, np.ones((3, 3), np.uint8))) * 255
        thr = lambda im, k: ((lambda m: (m > 0.3 * m.max()))(np.hypot(*gradient(im, k)))).astype(np.uint8)  # noqa: E731
        print(f"F1 к эталону без шума: Собель {common.f1(edge, thr(g, 'sobel'), 2):.3f}, Канни {common.f1(edge, mine, 2):.3f}; "
              f"шум s=20: Робертс {common.f1(edge, thr(noisy, 'roberts'), 2):.3f}, Собель {common.f1(edge, thr(noisy, 'sobel'), 2):.3f}, "
              f"Канни {common.f1(edge, canny(noisy, 2.0), 2):.3f}")
    mag, lap = np.hypot(gx, gy), np.abs(correlate(g, LAPLACE))
    common.grid([(g, "исходное"), (u8(mag / mag.max() * 255), "|grad| Собеля"), (u8(lap / lap.max() * 255), "Лапласиан"),
                 (mine, "Канни (наш)"), (ref, "cv2.Canny"), (canny(noisy, 2.0), "Канни, шум s=20")], args.out, "lab11.png")


if __name__ == "__main__":
    main()
