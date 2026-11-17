"""Лабораторная работа 13. Математическая морфология Серра.

Структурирующий элемент (СЭ) B «ощупывает» изображение A:
  эрозия     (A ⊖ B)(x) = min_{b in B} A(x + b)    -- сжатие, убирает мелкое светлое;
  дилатация  (A ⊕ B)(x) = max_{b in B} A(x - b)    -- расширение, заполняет мелкое тёмное;
  открытие   A ∘ B = (A ⊖ B) ⊕ B   -- убирает объекты меньше B, форму крупных сохраняет;
  закрытие   A • B = (A ⊕ B) ⊖ B   -- заполняет дыры и разрывы меньше B.
Для бинарных изображений min/max превращаются в И/ИЛИ. Идемпотентность:
(A ∘ B) ∘ B = A ∘ B. Нормализация фона (методичка): два СЭ -- малый (против
шума) и большой (подавляет объект), разность открытий выделяет объекты.

Запуск: python lab13_morphology_serra.py --demo | --image файл.png
"""

import sys

import cv2
import numpy as np

import common


def structuring_element(shape="rect", size=3):
    """Структурирующий элемент: булев массив size x size (size нечётный)."""
    if size < 1 or size % 2 == 0:
        raise ValueError("Размер СЭ должен быть нечётным >= 1")
    cv_shape = {"rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS,
                "ellipse": cv2.MORPH_ELLIPSE}[shape]
    return cv2.getStructuringElement(cv_shape, (size, size)).astype(bool)


def erode(img, se, border_value=255):
    """Эрозия: минимум по точкам СЭ.

    За краем изображения подставляется border_value: 255 («бесконечность»)
    не стирает объекты, прилегающие к границе -- так делает и OpenCV.
    """
    common.check_gray_u8(img)
    ph, pw = se.shape[0] // 2, se.shape[1] // 2
    h, w = img.shape
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant", constant_values=border_value)
    out = np.full_like(img, 255)
    for dy, dx in zip(*np.nonzero(se)):          # цикл по точкам СЭ, а не по пикселям
        np.minimum(out, padded[dy:dy + h, dx:dx + w], out=out)
    return out


def dilate(img, se, border_value=0):
    """Дилатация: максимум по ОТРАЖЁННОМУ СЭ (для симметричных СЭ -- то же самое)."""
    common.check_gray_u8(img)
    ph, pw = se.shape[0] // 2, se.shape[1] // 2
    h, w = img.shape
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant", constant_values=border_value)
    out = np.zeros_like(img)
    for dy, dx in zip(*np.nonzero(se[::-1, ::-1])):
        np.maximum(out, padded[dy:dy + h, dx:dx + w], out=out)
    return out


def opening(img, se):
    """Открытие: эрозия, затем дилатация."""
    return dilate(erode(img, se), se)


def closing(img, se):
    """Закрытие: дилатация, затем эрозия."""
    return erode(dilate(img, se), se)


def gradient(img, se):
    """Морфологический градиент: дилатация минус эрозия (толщина границы ~ размер СЭ)."""
    return dilate(img, se) - erode(img, se)


def top_hat(img, se):
    """Белый top-hat: img - открытие (светлые детали меньше СЭ)."""
    return img - opening(img, se)


def black_hat(img, se):
    """Чёрный top-hat: закрытие - img (тёмные детали меньше СЭ)."""
    return closing(img, se) - img


def background_normalization(gray, se_small, se_big):
    """Нормализация фона по Серра для светлых объектов.

    1. smooth = открытие(закрытие(img, B1), B1): малый СЭ убирает импульсы ОБОИХ
       знаков. Одного открытия мало: тёмный импульс при эрозии большим СЭ
       «затемнил» бы всё окно B2 и карта фона обнулилась бы.
    2. background = открытие(smooth, B2): большой СЭ срезает объекты, оставляя фон.
    3. объекты = smooth - background (>= 0: открытие не увеличивает яркость).
    """
    smooth = opening(closing(gray, se_small), se_small)
    background = opening(smooth, se_big)
    return smooth - background, background


def hit_or_miss(binary, se_fg, se_bg):
    """Преобразование «попал/промахнулся»: найти шаблон (se_fg -- объект, se_bg -- фон).

    HMT(A) = (A ⊖ B1) ∩ (A^c ⊖ B2): оба условия должны выполняться одновременно.
    """
    a_fg = erode(binary, se_fg, border_value=0)
    a_bg = erode(255 - binary, se_bg, border_value=0)
    return np.minimum(a_fg, a_bg)


def skeleton(binary, se=None):
    """Скелет по Лантуэжулю: объединение (A ⊖ kB) \\ ((A ⊖ kB) ∘ B) по k = 0, 1, ...

    Каждый член -- «то, что исчезло бы при открытии на k-м шаге».
    """
    se = structuring_element("cross", 3) if se is None else se
    skel = np.zeros_like(binary)
    cur = binary.copy()
    for _ in range(max(binary.shape)):     # конечное число шагов: объект исчезает при эрозии
        if not cur.any():
            break
        op = dilate(erode(cur, se, 0), se)
        skel |= cur & ~op
        cur = erode(cur, se, 0)
    return skel


def main():
    """Точка входа."""
    args = common.build_parser(__doc__).parse_args()
    if not (args.demo or args.image):
        print("Нужен --demo или --image")
        return 1
    if args.image:
        gray, truth = common.to_gray(common.load_image(args.image)), None
    else:
        gray, truth = common.demo_objects(noise=4.0)
    common.ensure_dir(args.out)

    # Сравнение с OpenCV: результаты должны совпадать точно (целочисленная арифметика).
    for shape in ("rect", "cross", "ellipse"):
        se = structuring_element(shape, 5)
        ku = se.astype(np.uint8)
        ok = all(np.array_equal(a, b) for a, b in [
            (erode(gray, se), cv2.erode(gray, ku)),
            (dilate(gray, se), cv2.dilate(gray, ku)),
            (opening(gray, se), cv2.morphologyEx(gray, cv2.MORPH_OPEN, ku)),
            (closing(gray, se), cv2.morphologyEx(gray, cv2.MORPH_CLOSE, ku)),
            (gradient(gray, se), cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, ku))])
        print(f"[сравнение с cv2, СЭ {shape} 5x5] erode/dilate/open/close/gradient совпадают: {ok}")

    se = structuring_element("ellipse", 5)
    o1 = opening(gray, se)
    print(f"[идемпотентность] (A∘B)∘B == A∘B: {np.array_equal(opening(o1, se), o1)}")

    # Время растёт линейно с числом точек СЭ.
    for size in (3, 9, 21):
        s = structuring_element("rect", size)
        print(f"[время] erode, СЭ {size}x{size}: {common.timeit(erode, gray, s, repeat=2) * 1e3:.1f} мс | "
              f"cv2.erode {common.timeit(cv2.erode, gray, s.astype(np.uint8)) * 1e3:.3f} мс")

    # Тест на шум (бинарное): «соль» убирает открытие, «перец» (дыры) -- закрытие.
    binary = (common.demo_objects()[1]).astype(np.uint8) * 255
    noisy = common.add_salt_pepper(binary, 0.04)
    se3 = structuring_element("rect", 3)
    cleaned = closing(opening(noisy, se3), se3)
    err = lambda img: int(np.count_nonzero((img > 0) != (binary > 0)))  # noqa: E731
    print(f"[шум 4%] ошибочных пикселей: шум {err(noisy)} -> после открытия+закрытия {err(cleaned)}")
    lonely = hit_or_miss(noisy, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], bool),
                         np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], bool))
    print(f"[hit-or-miss] найдено изолированных пикселей-«соли»: {int((lonely > 0).sum())}")

    # Нормализация фона по Серра на объектах с градиентом освещения.
    img_bg, truth_bg = common.demo_objects(noise=6.0)
    objects, background = background_normalization(
        common.add_salt_pepper(img_bg, 0.02), structuring_element("rect", 3),
        structuring_element("rect", 41))
    print(f"[нормализация фона] IoU (порог 40) = {common.iou(objects > 40, truth_bg):.3f}")

    skel = skeleton(binary)
    print(f"[скелет] пикселей объекта {int((binary > 0).sum())} -> скелет {int((skel > 0).sum())}")

    common.save_grid([(binary, "бинарное"), (noisy, "бинарное + шум 4%"),
                      (cleaned, "открытие + закрытие"), (skel, "скелет Лантуэжуля"),
                      (gradient(binary, se3), "морфол. градиент"), (objects, "нормализация фона (Серра)"),
                      (top_hat(img_bg, structuring_element("rect", 41)), "top-hat"),
                      (black_hat(img_bg, structuring_element("rect", 41)), "black-hat")],
                     f"{args.out}/lab13_morphology.png", ncols=4)
    print(f"Графики: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
