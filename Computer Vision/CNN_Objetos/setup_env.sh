#!/usr/bin/env bash
set -e

DEFAULT_PYTHON="/home/renato/Desktop/5to/IA/Codigos-IA/Iris/.uv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"

"$PYTHON_BIN" - <<'PY'
import sys

major, minor = sys.version_info[:2]
if (major, minor) >= (3, 14):
    raise SystemExit(
        "TensorFlow no tiene wheel compatible para este entorno con Python "
        f"{major}.{minor}. Usa Python 3.12, por ejemplo:\n"
        "PYTHON_BIN=python3.12 ./setup_env.sh"
    )
PY

"$PYTHON_BIN" -m venv --clear .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Entorno listo. Activalo con: source .venv/bin/activate"
