"""Лабораторная работа 12. Выделение контуров на бинарных изображениях.

Два подхода:
1. Граничные пиксели: объект минус его эрозия (B = A - (A erode cross)).
   Дают набор пикселей, но не упорядоченную кривую.
2. Прослеживание контура (Moore-neighbor tracing): обход границы по
   часовой стрелке, результат -- упорядоченная замкнутая цепочка точек,
   из которой считаются цепной код Фримана, периметр, площадь.

Запуск: python lab12_binary_contours.py --demo | --image файл.png
"""

import sys

import cv2
import numpy as np

import common
from lab09_binary_segmentation import label_components

# Направления (dx, dy) по часовой стрелке на экране (ось y вниз):
# 0=W, 1=NW, 2=N, 3=NE, 4=E, 5=SE, 6=S, 7=SW.
DIRS = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
# Цепной код Фримана -- стандартная нумерация против часовой, от E (0 = восток).
FREEMAN = {(1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
           (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7}


def boundary_pixels(binary, connectivity=4):
    """Внутренняя граница: пиксели объекта, у которых есть соседи-фон.

    Эрозия 4-связным крестом даёт 8-связную границу (тонкую, замкнутую);
    эрозия 3x3 -- 4-связную границу (толще).
    """
    common.check_gray_u8(binary)
    fg = binary > 0
    p = np.pad(fg, 1, mode="constant", constant_values=False)
    h, w = fg.shape
    inner = p[1:h + 1, 1:w + 1] & p[0:h, 1:w + 1] & p[2:h + 2, 1:w + 1] \
        & p[1:h + 1, 0:w] & p[1:h + 1, 2:w + 2]
    if connectivity == 8:
        inner &= p[0:h, 0:w] & p[0:h, 2:w + 2] & p[2:h + 2, 0:w] & p[2:h + 2, 2:w + 2]
    return (fg & ~inner).astype(np.uint8) * 255


def moore_trace(binary, start):
    """Прослеживание внешнего контура методом Мура с критерием остановки Джейкоба.

    Args:
        binary: 2D uint8 (объект ненулевой).
        start: (x, y) самого верхнего левого пикселя объекта: слева от него
            заведомо фон, это начальное направление «возврата».

    Returns:
        Список (x, y) -- упорядоченный контур по часовой стрелке.
    """
    common.check_gray_u8(binary)
    pad = np.pad(binary > 0, 1, mode="constant")  # рамка из фона: проверки границ не нужны
    sx, sy = start[0] + 1, start[1] + 1
    if not pad[sy, sx]:
        raise ValueError("Стартовая точка не принадлежит объекту")
    p, d = (sx, sy), 0           # d -- направление от p к «пикселю возврата» (фону)
    contour, first_move = [p], None
    max_steps = 8 * int(pad.sum()) + 8   # страховка от зацикливания
    for _ in range(max_steps):
        for k in range(1, 9):    # обход соседей по часовой стрелке, начиная после «возврата»
            nd = (d + k) % 8
            q = (p[0] + DIRS[nd][0], p[1] + DIRS[nd][1])
            if pad[q[1], q[0]]:
                break
        else:
            break                # соседей нет: изолированный пиксель
        if first_move is None:
            first_move = q
        elif p == (sx, sy) and q == first_move:
            break                # критерий Джейкоба: снова вышли из старта тем же ходом
        prev = (p[0] + DIRS[(nd - 1) % 8][0], p[1] + DIRS[(nd - 1) % 8][1])  # фон перед q
        d = DIRS.index((prev[0] - q[0], prev[1] - q[1]))
        p = q
        contour.append(p)
    if len(contour) > 1 and contour[-1] == contour[0]:
        contour.pop()
    return [(x - 1, y - 1) for x, y in contour]


def outer_contours(binary, connectivity=8):
    """Внешний контур каждой связной компоненты (по методу Мура)."""
    labels, n = label_components(binary, connectivity)
    # Первое вхождение метки в растровом порядке -- её верхний левый пиксель (O(N) за раз).
    _, first = np.unique(labels.ravel(), return_index=True)
    w = labels.shape[1]
    return [moore_trace(binary, (int(idx % w), int(idx // w))) for idx in first[1:]]


def chain_code(contour):
    """Цепной код Фримана замкнутого контура (8 направлений)."""
    pts = contour + contour[:1]
    return [FREEMAN[(b[0] - a[0], b[1] - a[1])] for a, b in zip(pts[:-1], pts[1:])
            if (b[0] - a[0], b[1] - a[1]) in FREEMAN]


def perimeter(contour):
    """Периметр: чётные коды (по сторонам) = 1, нечётные (диагональ) = sqrt(2)."""
    return sum(1.0 if c % 2 == 0 else np.sqrt(2.0) for c in chain_code(contour))


def polygon_area(contour):
    """Площадь многоугольника по формуле Гаусса (шнуровки): 1/2 |SUM x_i y_{i+1} - x_{i+1} y_i|.

    Считается по центрам пикселей контура, поэтому меньше числа пикселей объекта.
    """
    pts = np.array(contour, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def main():
    """Точка входа."""
    args = common.build_parser(__doc__).parse_args()
    if not (args.demo or args.image):
        print("Нужен --demo или --image")
        return 1
    if args.image:
        gray = common.to_gray(common.load_image(args.image))
        binary = (gray > cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[0]).astype(np.uint8) * 255
    else:
        binary = common.demo_objects()[1].astype(np.uint8) * 255
    common.ensure_dir(args.out)

    b4 = boundary_pixels(binary, 4)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # Край кадра -- фон (как и у прослеживания), поэтому borderValue=0.
    ref_b = binary - cv2.erode(binary, cross, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    print(f"[граничные пиксели] max|ours - (A - erode)| = "
          f"{int(np.abs(b4.astype(int) - ref_b.astype(int)).max())}")

    contours = outer_contours(binary)
    # RETR_CCOMP + «нет родителя» = внешние границы ВСЕХ компонент, включая острова внутри
    # дыр (RETR_EXTERNAL их пропускает, а наша разметка трассирует каждую компоненту).
    all_c, hier = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv_contours = [c for c, hh in zip(all_c, hier[0]) if hh[3] == -1]
    ours_set = {pt for c in contours for pt in c}
    ref_set = {(int(p[0][0]), int(p[0][1])) for c in cv_contours for p in c}
    print(f"[Мур vs cv2.findContours] контуров: {len(contours)} / {len(cv_contours)}; "
          f"множества точек совпадают: {ours_set == ref_set}")

    labels, _ = label_components(binary, 8)
    pixel_count = np.bincount(labels.ravel())
    # Каждая точка контура принадлежит ровно одному контуру cv2 -> сопоставление по словарю.
    point_to_cv = {(int(p[0][0]), int(p[0][1])): k for k, c in enumerate(cv_contours) for p in c}
    shown = min(len(contours), 10)
    print(f"[свойства первых {shown} контуров]  №  точек  периметр  cv2.arcLength  "
          "площадь(контур)  cv2.contourArea  пикселей")
    for i, c in enumerate(contours[:shown], start=1):
        match = cv_contours[point_to_cv[c[0]]]
        print(f"           {i:2d} {len(c):6d} {perimeter(c):9.1f} {cv2.arcLength(match, True):13.1f}"
              f" {polygon_area(c):15.1f} {cv2.contourArea(match):16.1f} {pixel_count[i]:9d}")

    print(f"[время] Мур {common.timeit(outer_contours, binary, repeat=2) * 1e3:.1f} мс | "
          f"cv2.findContours "
          f"{common.timeit(cv2.findContours, binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) * 1e3:.3f} мс")

    # Тест на шум: периметр катастрофически чувствителен к шуму.
    noisy = common.add_salt_pepper(binary, 0.02)
    cleaned = cv2.medianBlur(noisy, 3)
    for name, img in (("чистое", binary), ("шум 2%", noisy), ("шум + медиана 3x3", cleaned)):
        cs = outer_contours(img)
        print(f"[периметры] {name:18s}: контуров {len(cs):4d}, суммарный периметр "
              f"{sum(perimeter(c) for c in cs):8.1f}")

    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for c in contours:
        for x, y in c:
            vis[y, x] = (0, 0, 255)
    common.save_grid([(binary, "бинарное"), (b4, "граничные пиксели (A - erode)"),
                      (vis, "контур по методу Мура"), (noisy, "бинарное + шум")],
                     f"{args.out}/lab12_contours.png", ncols=2)
    print(f"Графики: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
