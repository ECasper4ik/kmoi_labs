"""Лаба 9. Сегментация бинарных изображений: разметка связных компонент (два прохода + union-find)."""
import cv2
import numpy as np

import common


def label(binary, conn=8):
    """Метки 1..n для компонент 4- или 8-связности -> (labels, n)."""
    common.check(binary)
    h, w = binary.shape
    labels = np.zeros((h, w), np.int32)
    parent = [0]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]  # сжатие путей
            a = parent[a]
        return a

    for y, x in zip(*map(np.ndarray.tolist, np.nonzero(binary))):
        # уже просмотренные соседи: слева, сверху (и две верхние диагонали при 8-связности)
        cand = [(y, x - 1), (y - 1, x)] + ([(y - 1, x - 1), (y - 1, x + 1)] if conn == 8 else [])
        nb = [labels[j, i] for j, i in cand if 0 <= i < w and j >= 0 and labels[j, i]]
        if not nb:
            parent.append(len(parent))
            labels[y, x] = len(parent) - 1
        else:
            labels[y, x] = m = min(nb)
            for v in nb:  # метки соседей -- один объект
                ra, rb = find(m), find(v)
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(len(parent))])
    uniq = np.unique(roots[1:])
    remap = np.zeros(len(parent), np.int32)
    remap[1:] = np.searchsorted(uniq, roots[1:]) + 1
    return remap[labels], len(uniq)


def properties(labels, n):
    """Площадь, центр масс (cx, cy) и bbox (x0, y0, x1, y1) каждой компоненты."""
    ys, xs = np.indices(labels.shape)
    f = labels.ravel()
    area = np.bincount(f, minlength=n + 1)[1:].astype(float)
    cx = np.bincount(f, xs.ravel(), n + 1)[1:] / np.maximum(area, 1)
    cy = np.bincount(f, ys.ravel(), n + 1)[1:] / np.maximum(area, 1)
    fg = labels > 0
    box = np.array([[labels.shape[1]] * (n + 1), [labels.shape[0]] * (n + 1), [-1] * (n + 1), [-1] * (n + 1)])
    np.minimum.at(box[0], labels[fg], xs[fg])
    np.minimum.at(box[1], labels[fg], ys[fg])
    np.maximum.at(box[2], labels[fg], xs[fg])
    np.maximum.at(box[3], labels[fg], ys[fg])
    return area, cx, cy, box[:, 1:].T


def drop_small(labels, n, min_area):
    """Удалить компоненты площадью < min_area (шум), перенумеровать."""
    keep = np.bincount(labels.ravel(), minlength=n + 1) >= min_area
    keep[0] = False
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, keep.sum() + 1)
    return remap[labels], int(keep.sum())


def main():
    args = common.parse(__doc__)
    if args.image:
        g = common.load(args.image)
        binary = (g > cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)[0]).astype(np.uint8) * 255
    else:
        binary = common.demo_objects()[1].astype(np.uint8) * 255
    for conn in (4, 8):
        lab, n = label(binary, conn)
        ref_n, ref = cv2.connectedComponents(binary, connectivity=conn)
        pairs = set(zip(lab[lab > 0].tolist(), ref[lab > 0].tolist()))
        print(f"{conn}-связность: наши {n}, cv2 {ref_n - 1}, разбиения совпадают: {len(pairs) == n == ref_n - 1}")
    diag = np.zeros((6, 6), np.uint8)
    diag[1, 1] = diag[2, 2] = diag[3, 3] = 255
    print(f"диагональ: 4-связность {label(diag, 4)[1]} объекта, 8-связность {label(diag, 8)[1]}")
    lab, n = label(binary)
    area, cx, cy, box = properties(lab, n)
    for i in range(min(n, 10)):
        print(f"  №{i + 1}: площадь {int(area[i])}, центр ({cx[i]:.1f}, {cy[i]:.1f}), bbox {box[i].tolist()}")
    print(f"время: наш {common.ms(label, binary, 8, repeat=1):.1f} мс, cv2 {common.ms(cv2.connectedComponents, binary):.3f} мс")
    noisy = common.salt_pepper(binary, 0.01)
    ln, nn = label(noisy)
    lf, nf = drop_small(ln, nn, 30)
    print(f"шум 1%: компонент {nn} (честных {n}), после фильтра площади >= 30: {nf}")
    color = np.random.default_rng(1).integers(60, 255, (nn + 1, 3)).astype(np.uint8)
    color[0] = 0
    common.grid([(binary, "бинарное"), (color[lab], f"метки n={n}"), (color[ln], f"шум: n={nn}"),
                 (color[lf], f"фильтр площади: n={nf}")], args.out, "lab09.png", 4)


if __name__ == "__main__":
    main()
