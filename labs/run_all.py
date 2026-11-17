"""Запуск всех 14 лабораторных в демо-режиме (или на своём изображении).

    python run_all.py                 # демо-данные, графики в out/
    python run_all.py --image my.png  # свой файл (для лаб, где он нужен)
"""

import argparse
import glob
import os
import subprocess
import sys


def main():
    """Последовательно запустить lab01..lab14 и показать, какие завершились успешно."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="путь к изображению (иначе --demo)")
    parser.add_argument("--out", default="out", help="папка для графиков")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    failed = []
    for path in sorted(glob.glob(os.path.join(here, "lab[0-9][0-9]_*.py"))):
        name = os.path.basename(path)
        cmd = [sys.executable, path, "--out", args.out]
        cmd += ["--image", args.image] if args.image else ["--demo"]
        print(f"\n===== {name} =====", flush=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
    print("\nИтог:", "все лабы выполнены" if not failed else f"ошибки в: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
