"""Лаба 12. Контуры бинарных изображений: граничные пиксели A - (A erode B) и прослеживание Мура."""
import cv2
import numpy as np

import common
from lab09_binary_segmentation import label

DIRS = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]  # W NW N NE E SE S SW (по часовой)


def boundary(binary):
    """Пиксели объекта, у которых есть 4-сосед из фона (край кадра -- фон)."""
    common.check(binary)
    p = np.pad(binary > 0, 1)
    h, w = binary.shape
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return ((binary > 0) & ~inner).astype(np.uint8) * 255


def moore(binary, start):
    """Контур по часовой стрелке из верхнего левого пикселя start = (x, y); стоп по критерию Джейкоба."""
    pad = np.pad(binary > 0, 1)
    start = (start[0] + 1, start[1] + 1)
    p, d, first, out = start, 0, None, [start]  # d: направление от p к «пикселю возврата» (фону)
    for _ in range(8 * int(pad.sum()) + 8):
        for k in range(1, 9):  # соседи по часовой стрелке, начиная после «возврата»
            nd = (d + k) % 8
            q = (p[0] + DIRS[nd][0], p[1] + DIRS[nd][1])
            if pad[q[1], q[0]]:
                break
        else:
            break  # изолированный пиксель
        if first is None:
            first = q
        elif p == start and q == first:
            break  # снова вышли из старта тем же ходом
        prev = (p[0] + DIRS[(nd - 1) % 8][0], p[1] + DIRS[(nd - 1) % 8][1])
        d = DIRS.index((prev[0] - q[0], prev[1] - q[1]))
        p = q
        out.append(p)
    if len(out) > 1 and out[-1] == start:
        out.pop()
    return [(x - 1, y - 1) for x, y in out]


def contours(binary):
    """Внешний контур каждой связной компоненты."""
    lab, _ = label(binary)
    _, first = np.unique(lab.ravel(), return_index=True)  # первое вхождение = верхний левый пиксель
    return [moore(binary, (int(i % lab.shape[1]), int(i // lab.shape[1]))) for i in first[1:]]


def perimeter(c):
    """Периметр по цепному коду: шаг по стороне = 1, по диагонали = sqrt(2)."""
    s = np.abs(np.diff(np.array(c + c[:1]), axis=0))
    return float(np.where(s.sum(axis=1) == 2, np.sqrt(2), 1).sum()) if len(c) > 1 else 0.0


def main():
    args = common.parse(__doc__)
    if args.image:
        g = common.load(args.image)
        binary = (g > cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)[0]).astype(np.uint8) * 255
    else:
        binary = common.demo_objects()[1].astype(np.uint8) * 255
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    ref_b = binary - cv2.erode(binary, cross, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    print("max|diff| граничных пикселей с A - erode:", int(np.abs(boundary(binary).astype(int) - ref_b).max()))
    cs = contours(binary)
    all_c, hier = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # внешние границы, включая острова в дырах
    cv_c = [c for c, hh in zip(all_c, hier[0]) if hh[3] == -1]
    same = {p for c in cs for p in c} == {(int(p[0][0]), int(p[0][1])) for c in cv_c for p in c}
    print(f"контуров: наши {len(cs)}, cv2 {len(cv_c)}; множества точек совпадают: {same}")
    idx = {(int(p[0][0]), int(p[0][1])): k for k, c in enumerate(cv_c) for p in c}
    for i, c in enumerate(cs[:5], 1):
        print(f"  №{i}: точек {len(c)}, периметр {perimeter(c):.1f}, cv2.arcLength {cv2.arcLength(cv_c[idx[c[0]]], True):.1f}")
    print(f"время: Мур {common.ms(contours, binary, repeat=1):.1f} мс, cv2.findContours {common.ms(cv2.findContours, binary, 0, 2):.3f} мс")
    noisy = common.salt_pepper(binary, 0.02)
    for name, im in (("чистое", binary), ("шум 2%", noisy), ("шум + медиана", cv2.medianBlur(noisy, 3))):
        c = contours(im)
        print(f"  {name}: контуров {len(c)}, суммарный периметр {sum(perimeter(x) for x in c):.0f}")
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for c in cs:
        for x, y in c:
            vis[y, x] = (0, 0, 255)
    common.grid([(binary, "бинарное"), (boundary(binary), "A - erode"), (vis, "Мур")], args.out, "lab12.png")


if __name__ == "__main__":
    main()
