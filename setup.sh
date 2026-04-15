#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
JAVA8_HOME_CANDIDATE="/usr/lib/jvm/java-1.8.0-openjdk-amd64"   # solo per CICFlowMeter

log() {
  printf "[setup] %s\n" "$1"
}

ensure_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    printf "[setup] ERROR: comando mancante: %s\n" "$cmd" >&2
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# System deps — sempre
# ---------------------------------------------------------------------------
ensure_cmd python3

# ---------------------------------------------------------------------------
# Python version check (>= 3.8)
# ---------------------------------------------------------------------------
PY_MINOR="$(python3 -c 'import sys; print(sys.version_info.minor)')"
PY_MAJOR="$(python3 -c 'import sys; print(sys.version_info.major)')"
if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 8 ) ]]; then
  printf "[setup] ERROR: Python >= 3.8 richiesto (trovato %s.%s)\n" "$PY_MAJOR" "$PY_MINOR" >&2
  exit 1
fi
log "Python ${PY_MAJOR}.${PY_MINOR} OK"

ensure_cmd sudo
ensure_cmd apt-get

# ---------------------------------------------------------------------------
# System deps — salta se già installati
# ---------------------------------------------------------------------------
SYSTEM_PKGS=(openjdk-8-jdk openjdk-17-jdk unrar p7zip-full)
MISSING_PKGS=()
for pkg in "${SYSTEM_PKGS[@]}"; do
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    MISSING_PKGS+=("$pkg")
  fi
done

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
  log "Installo dipendenze di sistema mancanti: ${MISSING_PKGS[*]}"
  sudo apt-get update -qq
  sudo apt-get install -y "${MISSING_PKGS[@]}"
else
  log "Dipendenze di sistema già installate, salto apt-get"
fi

# ---------------------------------------------------------------------------
# Java: Java 17 come JAVA_HOME di default (richiesto da PySpark, class file 61.0)
#       Java 8 rimane disponibile per CICFlowMeter tramite JAVA8_HOME in settings.py
# ---------------------------------------------------------------------------
JAVA17_CANDIDATE="/usr/lib/jvm/java-17-openjdk-amd64"
if [[ -d "$JAVA17_CANDIDATE" ]]; then
  export JAVA_HOME="$JAVA17_CANDIDATE"
  log "JAVA_HOME (Java 17) impostato su ${JAVA_HOME}"
else
  JAVA17_FOUND="$(update-java-alternatives --list 2>/dev/null | awk '/java-17/{print $3; exit}' || true)"
  if [[ -n "$JAVA17_FOUND" ]]; then
    export JAVA_HOME="$JAVA17_FOUND"
    log "JAVA_HOME (Java 17 fallback) impostato su ${JAVA_HOME}"
  else
    printf "[setup] ERROR: Java 17 non trovato dopo l'installazione.\n" >&2
    exit 1
  fi
fi

# Verifica che Java 8 esista ancora per CICFlowMeter
if [[ ! -d "$JAVA8_HOME_CANDIDATE" ]]; then
  printf "[setup] WARNING: Java 8 non trovato in %s — CICFlowMeter potrebbe non funzionare.\n" "$JAVA8_HOME_CANDIDATE"
fi

export PATH="${JAVA_HOME}/bin:${PATH}"

ensure_cmd java
JAVA_VER="$(java -version 2>&1 | head -1)"
log "Java: ${JAVA_VER}"

# ---------------------------------------------------------------------------
# Python venv — saltato se ensurepip non disponibile (es. Google Colab)
# ---------------------------------------------------------------------------
ensure_cmd python3

VENV_ACTIVE=false

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  log "Creo virtual environment in ${VENV_DIR}"
  if python3 -m venv "$VENV_DIR" 2>/dev/null; then
    log "Virtual environment creato"
  else
    log "WARNING: creazione venv fallita (ensurepip non disponibile?) — uso Python di sistema"
    rm -rf "$VENV_DIR"
  fi
fi

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  VENV_ACTIVE=true
  log "Virtual environment attivato"
else
  log "Uso Python di sistema: $(python3 --version)"
fi

log "Aggiorno pip/setuptools/wheel"
pip install -U pip setuptools wheel

log "Installo requirements"
pip install -r "${ROOT_DIR}/requirements.txt"

# ---------------------------------------------------------------------------
# Verifica moduli Python
# ---------------------------------------------------------------------------
log "Verifica moduli Python principali"
JAVA_HOME="$JAVA_HOME" python3 - <<'PY'
import importlib, sys
missing = []
for name in ["boto3", "pandas", "sklearn", "yaml", "pyspark"]:
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
if missing:
    print(f"[setup] ERROR: moduli mancanti: {missing}", file=sys.stderr)
    sys.exit(1)

# Smoke-test PySpark con JAVA_HOME già impostato
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[1]").appName("setup-check").getOrCreate()
spark.stop()
print("[setup] Python dependencies OK (PySpark smoke-test passato)")
PY

# ---------------------------------------------------------------------------
# Riepilogo
# ---------------------------------------------------------------------------
log "Setup completato"
log "Per eseguire: source .venv/bin/activate && python main.py"
