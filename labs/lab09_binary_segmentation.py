"""Лабораторная работа 9. Сегментация бинарных изображений.

Сегментация бинарного изображения = разметка связных компонент: каждому
объекту присваивается своя метка. Связность: 4 (соседи по сторонам) или
8 (ещё и по диагонали). Двухпроходный алгоритм: первый проход назначает
предварительные метки и фиксирует эквивалентности (union-find), второй --
заменяет метки на представителей классов. Сложность ~ O(N alpha(N)).

Запуск: python lab09_binary_segmentation.py --demo | --image файл.png
"""

import sys

import cv2
import numpy as np

import common


def label_components(binary, connectivity=8):
    """Разметка связных компонент двухпроходным алгоритмом с union-find.

    Args:
        binary: 2D uint8, объект = ненулевые пиксели.
        connectivity: 4 или 8.

    Returns:
        (labels int32 с метками 1..n, n). Фон = 0.
    """
    common.check_gray_u8(binary)
    if connectivity not in (4, 8):
        raise ValueError("connectivity должна быть 4 или 8")
    fg = binary > 0
    h, w = fg.shape
    labels = np.zeros((h, w), np.int32)
    parent = [0]  # parent[i] -- родитель метки i в лесе эквивалентностей

    def find(a):
        """Корень класса эквивалентности со сжатием путей."""
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(a, b):
        """Объединить классы; корнем остаётся меньшая метка."""
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    ys, xs = np.nonzero(fg)  # пиксели объектов в растровом порядке
    for y, x in zip(ys.tolist(), xs.tolist()):
        # Уже просмотренные соседи: слева, сверху (и две верхние диагонали при 8-связности).
        neigh = []
        if x > 0 and labels[y, x - 1]:
            neigh.append(int(labels[y, x - 1]))
        if y > 0:
            if labels[y - 1, x]:
                neigh.append(int(labels[y - 1, x]))
            if connectivity == 8:
                if x > 0 and labels[y - 1, x - 1]:
                    neigh.append(int(labels[y - 1, x - 1]))
                if x < w - 1 and labels[y - 1, x + 1]:
                    neigh.append(int(labels[y - 1, x + 1]))
        if not neigh:
            parent.append(len(parent))           # новая метка
            labels[y, x] = len(parent) - 1
        else:
            m = min(neigh)
            labels[y, x] = m
            for n in neigh:                      # фиксируем: все эти метки -- один объект
                union(m, n)

    # Второй проход: метка -> представитель класса -> последовательный номер 1..n.
    roots = np.array([find(i) for i in range(len(parent))])
    unique_roots = np.unique(roots[1:]) if len(parent) > 1 else np.array([], int)
    remap = np.zeros(len(parent), np.int32)
    for new, root in enumerate(unique_roots, start=1):
        remap[roots == root] = new
    return remap[labels], int(unique_roots.size)


def region_properties(labels, n):
    """Площадь, центр масс и ограничивающий прямоугольник каждой компоненты.

    Returns:
        dict с массивами длины n: area, cx, cy, bbox (x0, y0, x1, y1).
    """
    ys, xs = np.indices(labels.shape)
    flat = labels.ravel()
    area = np.bincount(flat, minlength=n + 1)[1:].astype(np.float64)
    safe = np.maximum(area, 1)
    cx = np.bincount(flat, weights=xs.ravel(), minlength=n + 1)[1:] / safe
    cy = np.bincount(flat, weights=ys.ravel(), minlength=n + 1)[1:] / safe
    fg = labels > 0
    x0 = np.full(n + 1, labels.shape[1])
    y0 = np.full(n + 1, labels.shape[0])
    x1 = np.full(n + 1, -1)
    y1 = np.full(n + 1, -1)
    np.minimum.at(x0, labels[fg], xs[fg])
    np.minimum.at(y0, labels[fg], ys[fg])
    np.maximum.at(x1, labels[fg], xs[fg])
    np.maximum.at(y1, labels[fg], ys[fg])
    return {"area": area, "cx": cx, "cy": cy,
            "bbox": np.stack([x0[1:], y0[1:], x1[1:], y1[1:]], axis=1)}


def filter_by_area(labels, n, min_area):
    """Удалить компоненты площадью < min_area (шум), метки перенумеруются."""
    area = np.bincount(labels.ravel(), minlength=n + 1)
    keep = area >= min_area
    keep[0] = False
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, keep.sum() + 1)
    return remap[labels], int(keep.sum())


def same_partition(a, b):
    """Совпадают ли разбиения (метки могут отличаться перестановкой номеров)."""
    fg = (a > 0) | (b > 0)
    if not np.array_equal(a > 0, b > 0):
        return False
    pairs = set(zip(a[fg].tolist(), b[fg].tolist()))
    return len({p[0] for p in pairs}) == len(pairs) == len({p[1] for p in pairs})


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

    for conn in (4, 8):
        labels, n = label_components(binary, conn)
        ref_n, ref = cv2.connectedComponents(binary, connectivity=conn)
        print(f"[{conn}-связность] наши компоненты: {n} | cv2: {ref_n - 1} | "
              f"разбиения совпадают: {same_partition(labels, ref)}")

    # Диагональное касание: 4-связность видит три отдельных пикселя, 8-связность -- один объект.
    diag = np.zeros((6, 6), np.uint8)
    diag[1, 1] = diag[2, 2] = diag[3, 3] = 255
    print(f"[диагональ] 4-связность: {label_components(diag, 4)[1]} объекта, "
          f"8-связность: {label_components(diag, 8)[1]}")

    labels, n = label_components(binary, 8)
    props = region_properties(labels, n)
    print("[свойства] №  площадь   центр масс       bbox")
    for i in range(n):
        print(f"           {i + 1:2d} {int(props['area'][i]):7d}  "
              f"({props['cx'][i]:6.1f}, {props['cy'][i]:6.1f})  {props['bbox'][i].tolist()}")

    print(f"[время] наш {common.timeit(label_components, binary, 8, repeat=2) * 1e3:.1f} мс | "
          f"cv2.connectedComponents {common.timeit(cv2.connectedComponents, binary, repeat=3) * 1e3:.3f} мс")

    # Тест на шум: «соль» создаёт ложные компоненты; фильтр по площади их убирает.
    noisy = common.add_salt_pepper(binary, 0.01)
    ln, nn = label_components(noisy, 8)
    lf, nf = filter_by_area(ln, nn, min_area=30)
    print(f"[шум 1%] компонент: {nn} (честных {n}); после фильтра площади >= 30: {nf}")

    palette = np.random.default_rng(1).integers(60, 255, (max(n, nn) + 1, 3)).astype(np.uint8)
    palette[0] = 0
    common.save_grid([(binary, "бинарное"), (palette[labels], f"метки (n={n})"),
                      (noisy, "бинарное + шум"), (palette[ln], f"метки с шумом (n={nn})"),
                      (palette[lf], f"после фильтра площади (n={nf})")],
                     f"{args.out}/lab09_labels.png", ncols=3)
    print(f"Графики: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
