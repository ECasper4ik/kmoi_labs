"""Лаба 13. Морфология Серра: эрозия (min), дилатация (max), открытие, закрытие, нормализация фона."""
import cv2
import numpy as np

import common


def se(shape="rect", size=3):
    """Структурирующий элемент (bool, size нечётный)."""
    kind = {"rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS, "ellipse": cv2.MORPH_ELLIPSE}[shape]
    return cv2.getStructuringElement(kind, (size, size)).astype(bool)


def _fold(img, s, func, fill):
    """Свёртка сдвигов изображения по точкам СЭ операцией min/max; за краем -- fill."""
    common.check(img)
    ph, pw = s.shape[0] // 2, s.shape[1] // 2
    h, w = img.shape
    pad = np.pad(img, ((ph, ph), (pw, pw)), constant_values=fill)
    out = np.full_like(img, 255 if func is np.minimum else 0)  # нейтральный элемент операции
    for dy, dx in zip(*np.nonzero(s)):
        out = func(out, pad[dy:dy + h, dx:dx + w])
    return out


def erode(img, s, fill=255):
    """(A erode B)(x) = min A(x + b); fill=255 -- край не «съедает» объекты."""
    return _fold(img, s, np.minimum, fill)


def dilate(img, s):
    """(A dilate B)(x) = max A(x - b): отражённый СЭ."""
    return _fold(img, s[::-1, ::-1], np.maximum, 0)


def opening(img, s):
    return dilate(erode(img, s), s)


def closing(img, s):
    return erode(dilate(img, s), s)


def background_normalization(gray, small, big):
    """open-close малым СЭ убирает шум обоих знаков; открытие большим -- фон; объекты = шум-фон."""
    smooth = opening(closing(gray, small), small)
    bg = opening(smooth, big)
    return smooth - bg, bg


def hit_or_miss(binary, s_fg, s_bg):
    """(A erode B1) AND (A^c erode B2): поиск шаблона."""
    return np.minimum(erode(binary, s_fg, 0), erode(255 - binary, s_bg, 0))


def skeleton(binary, s=None):
    """Скелет Лантуэжуля: объединение (A erode kB) \\ ((A erode kB) open B)."""
    s = se("cross") if s is None else s
    skel, cur = np.zeros_like(binary), binary.copy()
    for _ in range(max(binary.shape)):
        if not cur.any():
            break
        skel |= cur & ~dilate(erode(cur, s, 0), s)
        cur = erode(cur, s, 0)
    return skel


def main():
    args = common.parse(__doc__)
    g = common.load(args.image) if args.image else common.demo_objects(noise=4)[0]
    for shape in ("rect", "cross", "ellipse"):
        s = se(shape, 5)
        u = s.astype(np.uint8)
        ok = all(np.array_equal(a, b) for a, b in [(erode(g, s), cv2.erode(g, u)), (dilate(g, s), cv2.dilate(g, u)),
                 (opening(g, s), cv2.morphologyEx(g, cv2.MORPH_OPEN, u)), (closing(g, s), cv2.morphologyEx(g, cv2.MORPH_CLOSE, u))])
        print(f"СЭ {shape} 5x5: erode/dilate/open/close совпадают с cv2: {ok}")
    o = opening(g, se("ellipse", 5))
    print("идемпотентность открытия:", np.array_equal(opening(o, se("ellipse", 5)), o))
    print("время erode 3/9/21:", [f"{common.ms(erode, g, se('rect', k), repeat=1):.1f} мс" for k in (3, 9, 21)])
    img, truth = common.demo_objects()
    binary = truth.astype(np.uint8) * 255
    noisy = common.salt_pepper(binary, 0.04)
    clean = closing(opening(noisy, se()), se())
    err = lambda a: int(np.count_nonzero((a > 0) != truth))  # noqa: E731
    print(f"шум 4%: ошибочных пикселей {err(noisy)} -> после открытия+закрытия {err(clean)}")
    objects, bg = background_normalization(common.salt_pepper(img, 0.02), se(), se("rect", 41))
    print(f"нормализация фона: IoU {common.iou(objects > 40, truth):.3f}; скелет: {int((skeleton(binary) > 0).sum())} пикселей")
    common.grid([(noisy, "шум 4%"), (clean, "открытие + закрытие"), (skeleton(binary), "скелет"),
                 (dilate(binary, se()) - erode(binary, se()), "градиент"), (bg, "фон"), (objects, "объекты")],
                args.out, "lab13.png")


if __name__ == "__main__":
    main()
