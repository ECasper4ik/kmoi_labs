"""Лаба 7. Линейная фильтрация: (I*K)(x,y) = sum K(i,j) I(x-i, y-j); корреляция без переворота ядра."""
import cv2
import numpy as np

import common

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
SOBEL_Y = SOBEL_X.T
LAPLACE = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)
SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float)  # I - Лапласиан


def box(k):
    """Усредняющее ядро: сумма весов 1 (яркость сохраняется)."""
    return np.full((k, k), 1.0 / k ** 2)


def gauss1d(k, sigma):
    x = np.arange(k) - (k - 1) / 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def correlate(img, kernel):
    """Корреляция (как cv2.filter2D); выход float. Цикл по ядру, а не по пикселям; края -- reflect."""
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Размер ядра должен быть нечётным")
    h, w = img.shape
    pad = np.pad(img.astype(float), ((kh // 2,) * 2, (kw // 2,) * 2), mode="reflect")
    return sum(kernel[i, j] * pad[i:i + h, j:j + w] for i in range(kh) for j in range(kw))


def convolve(img, kernel):
    """Истинная свёртка: корреляция с ядром, повёрнутым на 180 градусов."""
    return correlate(img, kernel[::-1, ::-1])


def gauss_separable(img, k, sigma):
    """Гаусс двумя 1D-проходами: O(N*2k) вместо O(N*k^2)."""
    g = gauss1d(k, sigma)
    return correlate(correlate(img, g[None]), g[:, None])


def u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def main():
    args = common.parse(__doc__)
    g = common.load(args.image) if args.image else common.demo_objects(noise=0)[0]
    gf, k = g.astype(np.float32), 5
    print("max|diff| с cv2: box", np.abs(correlate(g, box(k)) - cv2.filter2D(gf, -1, box(k).astype(np.float32), borderType=4)).max(),
          "| Sobel", np.abs(correlate(g, SOBEL_X) - cv2.filter2D(gf, -1, SOBEL_X.astype(np.float32), borderType=4)).max())
    print("свёртка с Собелем отличается знаком:", np.allclose(convolve(g, SOBEL_X), -correlate(g, SOBEL_X)))
    print("Гаусс разделимо == 2D:", np.allclose(gauss_separable(g, k, 1.0), correlate(g, np.outer(gauss1d(k, 1), gauss1d(k, 1)))))
    for kk in (3, 15):
        g2 = np.outer(gauss1d(kk, kk / 5), gauss1d(kk, kk / 5))
        print(f"время k={kk}: 2D {common.ms(correlate, g, g2):.1f} мс, разделимо {common.ms(gauss_separable, g, kk, kk / 5):.1f} мс, "
              f"cv2 {common.ms(cv2.GaussianBlur, g, (kk, kk), kk / 5):.3f} мс")
    noisy = common.gauss(g, 20)
    print(f"PSNR при шуме s=20: шум {common.psnr(g, noisy):.1f}, box5 {common.psnr(g, u8(correlate(noisy, box(5)))):.1f}, "
          f"gauss5 {common.psnr(g, u8(gauss_separable(noisy, 5, 1.0))):.1f} дБ")
    mag = np.hypot(correlate(g, SOBEL_X), correlate(g, SOBEL_Y))
    lap = np.abs(correlate(g, LAPLACE))
    common.grid([(g, "исходное"), (u8(correlate(g, box(k))), "усреднение 5x5"), (u8(gauss_separable(g, k, 1.0)), "Гаусс"),
                 (u8(correlate(g, SHARPEN)), "резкость"), (u8(mag / mag.max() * 255), "Собель |grad|"),
                 (u8(lap / lap.max() * 255), "Лапласиан")], args.out, "lab07.png")


if __name__ == "__main__":
    main()
