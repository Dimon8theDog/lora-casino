#!/bin/sh
# LoRA Casino on its own (Mac, Linux): starts its small server and opens the page in your browser.
# Start ComfyUI too (before or after): the pictures are drawn through it. Ctrl+C stops it.
cd "$(dirname "$0")" || exit 1

ok() { "$1" -c "import sys, numpy; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1; }

PY=""
for p in .venv/bin/python "${PYTHON:-}" python3 python; do
  [ -n "$p" ] && ok "$p" && PY="$p" && break
done

if [ -z "$PY" ]; then  # a Python without numpy: give LoRA Casino its own small environment (once)
  BASE=""
  for p in "${PYTHON:-}" python3 python; do
    [ -n "$p" ] && "$p" -c "import sys, venv; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1 && BASE="$p" && break
  done
  if [ -z "$BASE" ]; then
    echo "LoRA Casino needs Python 3.10 or newer with numpy. Install Python, then run this again."
    exit 1
  fi
  echo "Setting up LoRA Casino (once): a private Python environment with numpy..."
  "$BASE" -m venv .venv && .venv/bin/python -m pip install --quiet -r requirements.txt || exit 1
  PY=.venv/bin/python
fi

exec "$PY" standalone.py 8777 --open
