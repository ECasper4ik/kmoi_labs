"""Запуск всех лаб подряд: python run_all.py [--image файл.png]."""
import glob
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
failed = []
for path in sorted(glob.glob(os.path.join(here, "lab[0-9][0-9]_*.py"))):
    print("\n==", os.path.basename(path), flush=True)
    if subprocess.run([sys.executable, path, *sys.argv[1:]]).returncode:
        failed.append(os.path.basename(path))
print("\nИтог:", f"ошибки: {failed}" if failed else "все лабы выполнены")
sys.exit(bool(failed))
