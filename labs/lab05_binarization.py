"""Лабораторная работа 5. Бинаризация.

Методы: глобальный порог, Оцу (максимум межклассовой дисперсии), ISODATA
(итеративный порог), локальный порог Саувола (по интегральному изображению).
Бинаризация по порогу -- «адаптивная», если порог берётся из гистограммы
(методичка: бимодальная гистограмма -> порог во впадине между модами).

Запуск: python lab05_binarization.py --demo | --image файл.png
"""

import sys

import cv2
import numpy as np

import common


def global_threshold(gray, t):
    """B(x, y) = 255, если I(x, y) > t, иначе 0."""
    common.check_gray_u8(gray)
    return (gray > t).astype(np.uint8) * 255


def otsu_threshold(gray):
    """Порог Оцу: t* = argmax sigma_b^2(t).

    sigma_b^2(t) = (mu_T * w(t) - mu(t))^2 / (w(t) * (1 - w(t))), где
    w(t) -- доля пикселей <= t, mu(t) -- накопленное среднее, mu_T -- общее среднее.
    Максимум межклассовой дисперсии эквивалентен минимуму внутриклассовой.
    """
    common.check_gray_u8(gray)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)                      # доля класса «фон» (<= t)
    mu = np.cumsum(p * np.arange(256))        # накопленное среднее
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b = np.zeros(256)
    valid = denom > 1e-12                     # на краях знаменатель 0 -> пропускаем
    sigma_b[valid] = (mu_t * omega[valid] - mu[valid]) ** 2 / denom[valid]
    return int(np.argmax(sigma_b))


def isodata_threshold(gray, eps=0.5, max_iter=100):
    """Итеративный порог: T <- (mean(I <= T) + mean(I > T)) / 2 до сходимости."""
    common.check_gray_u8(gray)
    t = float(gray.mean())
    for _ in range(max_iter):
        low, high = gray[gray <= t], gray[gray > t]
        if low.size == 0 or high.size == 0:  # однородное изображение
            break
        t_new = 0.5 * (low.mean() + high.mean())
        if abs(t_new - t) < eps:
            t = t_new
            break
        t = t_new
    return int(round(t))


def _box_sums(img, win):
    """Суммы по окнам win x win через интегральное изображение, O(N) независимо от win."""
    r = win // 2
    padded = np.pad(img.astype(np.float64), r, mode="reflect")
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1))
    integral[1:, 1:] = padded.cumsum(0).cumsum(1)
    h, w = img.shape
    # S(window) = II[y+win, x+win] - II[y, x+win] - II[y+win, x] + II[y, x]
    return (integral[win:win + h, win:win + w] - integral[0:h, win:win + w]
            - integral[win:win + h, 0:w] + integral[0:h, 0:w])


def sauvola_threshold(gray, win=31, k=0.2, dyn=128.0, bright_objects=True):
    """Локальный порог Саувола: T = m * (1 + k * (s / R - 1)).

    m, s -- среднее и СКО в окне, R = 128 -- динамический диапазон СКО.
    Метод рассчитан на ТЁМНЫЙ объект на светлом фоне (объект: I <= T);
    для светлых объектов изображение предварительно инвертируется.
    Окно должно быть заметно больше объекта (>= 2x), иначе внутри
    крупного объекта m ~ I и он «дырявится».
    """
    common.check_gray_u8(gray)
    if win % 2 == 0 or win < 3:
        raise ValueError("win должно быть нечётным >= 3")
    work = 255 - gray if bright_objects else gray
    area = float(win * win)
    mean = _box_sums(work, win) / area
    sq_mean = _box_sums(work.astype(np.float64) ** 2, win) / area
    std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0.0))  # D = E[I^2] - E[I]^2 >= 0
    thr = mean * (1.0 + k * (std / dyn - 1.0))
    return (work <= thr).astype(np.uint8) * 255


def main():
    """Точка входа: сравнение методов на неравномерном освещении и шуме."""
    args = common.build_parser(__doc__).parse_args()
    if not (args.demo or args.image):
        print("Нужен --demo или --image")
        return 1
    if args.image:
        gray, truth = common.to_gray(common.load_image(args.image)), None
    else:
        gray, truth = common.demo_objects()
    common.ensure_dir(args.out)

    t_otsu = otsu_threshold(gray)
    t_cv, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"[Оцу] наш порог {t_otsu} | cv2 {int(t_cv)} (должны совпасть до +-1)")
    t_iso = isodata_threshold(gray)
    print(f"[ISODATA] порог {t_iso}")

    b_otsu = global_threshold(gray, t_otsu)
    b_loc = sauvola_threshold(gray, win=61)
    b_cv_ad = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 61, -10)
    if truth is not None:
        print(f"[IoU] Оцу {common.iou(b_otsu > 0, truth):.3f} | "
              f"Саувола {common.iou(b_loc > 0, truth):.3f} | "
              f"cv2.adaptiveThreshold {common.iou(b_cv_ad > 0, truth):.3f}")

    print(f"[время] Оцу {common.timeit(otsu_threshold, gray) * 1e3:.3f} мс | "
          f"Саувола {common.timeit(sauvola_threshold, gray, 61) * 1e3:.2f} мс | "
          f"cv2.threshold {common.timeit(cv2.threshold, gray, 0, 255, cv2.THRESH_OTSU) * 1e3:.3f} мс")

    noisy = common.add_gaussian_noise(gray, 25.0)
    filtered = cv2.medianBlur(noisy, 5)
    b_noisy = sauvola_threshold(noisy, win=61)
    b_noisy_f = sauvola_threshold(filtered, win=61)
    if truth is not None:
        print(f"[шум s=25] IoU Саувола: без фильтра {common.iou(b_noisy > 0, truth):.3f}, "
              f"после медианы {common.iou(b_noisy_f > 0, truth):.3f}")

    common.save_grid([(gray, "исходное"), (b_otsu, f"Оцу t={t_otsu}"),
                      (b_loc, "Саувола (локальный)"), (b_cv_ad, "cv2.adaptiveThreshold"),
                      (b_noisy, "Саувола + шум"), (b_noisy_f, "медиана + Саувола")],
                     f"{args.out}/lab05_binarization.png", ncols=3)
    print(f"Графики: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
