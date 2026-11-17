"""Лаба 14. Решение задачи (шаблон: текст задачи неизвестен): найти и измерить объекты на шумном изображении.

Конвейер: медиана -> нормализация фона -> Оцу -> открытие/закрытие -> разметка -> фильтр площади.
"""
import cv2
import numpy as np

import common
from lab05_binarization import otsu, threshold
from lab06_nonlinear_filtering import background_normalization, median
from lab09_binary_segmentation import drop_small, label, properties
from lab13_morphology_serra import closing, opening, se


def solve(gray, min_area=40, k_big=35):
    """Изображение светлых объектов -> (метки, число объектов, этапы)."""
    common.check(gray)
    filtered = median(gray, 3)
    normalized, _ = background_normalization(filtered, 1, k_big)  # окно фона > крупнейшего объекта
    binary = threshold(normalized, otsu(normalized))
    cleaned = closing(opening(binary, se("ellipse")), se("ellipse"))
    labels, n = drop_small(*label(cleaned), min_area)
    return labels, n, [(gray, "исходное"), (normalized, "нормализация фона"), (binary, "бинарное"), (cleaned, "морфология")]


def main():
    args = common.parse(__doc__, [("--min-area", {"type": int, "default": 40})])
    if args.image:
        g, truth = common.load(args.image), None
    else:
        base, truth = common.demo_objects(noise=10)
        g = common.salt_pepper(base, 0.03)
    labels, n, stages = solve(g, args.min_area)
    area, cx, cy, box = properties(labels, n)
    print(f"найдено объектов: {n}")
    for i in range(n):
        print(f"  №{i + 1}: площадь {int(area[i])}, центр ({cx[i]:.1f}, {cy[i]:.1f}), bbox {box[i].tolist()}")
    if truth is not None:
        print(f"эталон: {label(truth.astype(np.uint8) * 255)[1]} объектов, IoU {common.iou(labels > 0, truth):.3f}")
    print(f"время: {common.ms(solve, g, args.min_area, repeat=1):.1f} мс")
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for i, (x0, y0, x1, y1) in enumerate(box, 1):
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1)
        cv2.putText(vis, str(i), (int(x0), max(int(y0) - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    common.grid(stages + [(vis, f"результат: {n}")], args.out, "lab14.png", 5)


if __name__ == "__main__":
    main()
