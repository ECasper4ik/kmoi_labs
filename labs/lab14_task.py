"""Лабораторная работа 14. Решение задачи (сквозной конвейер).

ВНИМАНИЕ: в списке заданий пункт 14 назван только «Решение задачи», текст
задачи в материалах отсутствует. Это шаблон типовой задачи курса:
«найти, посчитать и измерить объекты на зашумлённом изображении с
неравномерной освещённостью». Конвейер собран из функций лаб 5, 6, 9, 13:

  1. медиана 3x3                       (лаб. 6)  -- убрать импульсный шум
  2. нормализация фона                 (лаб. 6)  -- убрать градиент освещения
  3. бинаризация по Оцу                (лаб. 5)
  4. открытие + закрытие               (лаб. 13) -- убрать остатки шума, заполнить дыры
  5. разметка связных компонент        (лаб. 9)
  6. фильтр по площади + измерения     (лаб. 9)

Когда будет известен текст задачи, замените этапы 3-6 на нужные.

Запуск: python lab14_task.py --demo | --image файл.png [--dark] [--min-area N]
"""

import sys
import time

import cv2
import numpy as np

import common
from lab05_binarization import global_threshold, otsu_threshold
from lab06_nonlinear_filtering import background_normalization, median_filter
from lab09_binary_segmentation import filter_by_area, label_components, region_properties
from lab13_morphology_serra import closing, opening, structuring_element


def solve(gray, min_area=40, bright_objects=True, k_big=35):
    """Конвейер: изображение -> (метки объектов, их число, словарь промежуточных этапов).

    Args:
        gray: 2D uint8.
        min_area: минимальная площадь объекта (меньшие -- шум).
        bright_objects: True -- объекты светлее фона, False -- темнее.
        k_big: апертура оценки фона; должна быть больше самого крупного объекта.
    """
    common.check_gray_u8(gray)
    stages, timings = {}, {}

    def step(name, func, *args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        timings[name] = (time.perf_counter() - t0) * 1e3
        return result

    stages["исходное"] = gray
    filtered = step("медиана 3x3", median_filter, gray, 3)
    stages["медиана"] = filtered
    normalized, _ = step("нормализация фона", background_normalization, filtered, 1, k_big,
                         bright_objects)
    stages["нормализация фона"] = normalized
    binary = step("бинаризация (Оцу)", lambda img: global_threshold(img, otsu_threshold(img)),
                  normalized)
    stages["бинарное"] = binary
    se = structuring_element("ellipse", 3)
    cleaned = step("морфология", lambda img: closing(opening(img, se), se), binary)
    stages["после морфологии"] = cleaned
    labels, n = step("разметка", label_components, cleaned, 8)
    labels, n = step("фильтр по площади", filter_by_area, labels, n, min_area)
    return labels, n, stages, timings


def main():
    """Точка входа."""
    parser = common.build_parser(__doc__)
    parser.add_argument("--dark", action="store_true", help="объекты темнее фона")
    parser.add_argument("--min-area", type=int, default=40, help="минимальная площадь объекта")
    args = parser.parse_args()
    if not (args.demo or args.image):
        print("Нужен --demo или --image")
        return 1
    if args.image:
        gray, truth = common.to_gray(common.load_image(args.image)), None
    else:
        base, truth = common.demo_objects(noise=10.0)
        gray = common.add_salt_pepper(base, 0.03)  # демо: шум + импульсный шум
    common.ensure_dir(args.out)

    labels, n, stages, timings = solve(gray, args.min_area, not args.dark)
    props = region_properties(labels, n)
    print(f"Найдено объектов: {n}")
    print(" №  площадь  центр (x, y)     bbox")
    for i in range(n):
        print(f"{i + 1:2d} {int(props['area'][i]):8d}  ({props['cx'][i]:6.1f}, {props['cy'][i]:6.1f})"
              f"  {props['bbox'][i].tolist()}")
    if truth is not None:
        true_n = label_components(truth.astype(np.uint8) * 255, 8)[1]
        print(f"[проверка] объектов в эталоне: {true_n}, найдено: {n}; "
              f"IoU маски = {common.iou(labels > 0, truth):.3f}")
    print("[время по этапам, мс] " + ", ".join(f"{k}: {v:.1f}" for k, v in timings.items()))

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i in range(n):
        x0, y0, x1, y1 = props["bbox"][i]
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1)
        cv2.putText(vis, str(i + 1), (int(x0), max(int(y0) - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1)
    items = [(img, name) for name, img in stages.items()] + [(vis, f"результат: {n} объектов")]
    common.save_grid(items, f"{args.out}/lab14_pipeline.png", ncols=3)
    print(f"Графики: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
