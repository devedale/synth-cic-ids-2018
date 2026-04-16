#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

log() {
  printf "[setup] %s\n" "$1"
}

ensure_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    printf "[setup] ERROR: missing command: %s\n" "$cmd" >&2
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# System deps — mandatory
# ---------------------------------------------------------------------------
ensure_cmd python3

# ---------------------------------------------------------------------------
# Python version check (>= 3.8)
# ---------------------------------------------------------------------------
PY_MINOR="$(python3 -c 'import sys; print(sys.version_info.minor)')"
PY_MAJOR="$(python3 -c 'import sys; print(sys.version_info.major)')"
if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 8 ) ]]; then
    printf "[setup] ERROR: Python >= 3.8 required (found %s.%s)\n" "$PY_MAJOR" "$PY_MINOR" >&2
    exit 1
fi
log "Python ${PY_MAJOR}.${PY_MINOR} OK"

# ---------------------------------------------------------------------------
# Python venv — skipped if ensurepip is unavailable (e.g. Google Colab)
# ---------------------------------------------------------------------------
VENV_ACTIVE=false

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  log "Creating virtual environment in ${VENV_DIR}"
  if python3 -m venv "$VENV_DIR" 2>/dev/null; then
    log "Virtual environment created"
  else
    log "WARNING: venv creation failed (ensurepip missing?) — using system Python"
    rm -rf "$VENV_DIR"
  fi
fi

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  VENV_ACTIVE=true
  log "Virtual environment activated"
else
  log "Using system Python: $(python3 --version)"
fi

log "Updating pip/setuptools/wheel"
pip install -U pip setuptools wheel

log "Installing requirements"
pip install -r "${ROOT_DIR}/requirements.txt"

# ---------------------------------------------------------------------------
# Verify exact python imports
# ---------------------------------------------------------------------------
log "Verifying Python package dependencies"
python3 - <<'PY'
import importlib, sys
missing = []
for name in ["boto3", "pandas", "sklearn", "numpy"]:
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
if missing:
    print(f"[setup] ERROR: missing modules: {missing}", file=sys.stderr)
    sys.exit(1)

print("[setup] Python dependencies OK")
PY

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "Setup for Synthetic Dataset Generator complete"
log "To run: source .venv/bin/activate && python main.py"
