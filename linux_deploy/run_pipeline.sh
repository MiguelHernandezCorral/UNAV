#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Lanzador del pipeline UNAV para Linux/Mac
# =============================================================================
# Uso:
#   bash run_pipeline.sh                          # todas las fases
#   bash run_pipeline.sh --phases fase3           # solo predicciones
#   bash run_pipeline.sh --dry-run                # sin escritura en Oracle/SF
#   bash run_pipeline.sh --phases fase1 fase2     # ingesta + limpieza
#   bash run_pipeline.sh --save-hist              # con historial de predicciones
#
# Requisitos previos:
#   - Python 3.11 con venv en .venv/  (ver docs/PIPELINE_DEV.md)
#   - Fichero .env con credenciales Oracle y Salesforce
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

# ── Verificaciones previas ────────────────────────────────────────────────────

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[ERROR] Virtual environment no encontrado en ${VENV_DIR}"
    echo "        Crea el entorno con: python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

if [[ ! -f "${SCRIPT_DIR}/.env" ]]; then
    echo "[WARN]  Fichero .env no encontrado — se usarán las variables de entorno del sistema."
fi

# ── Crear directorio de logs ──────────────────────────────────────────────────
mkdir -p "${LOG_DIR}"

# ── Activar entorno virtual ───────────────────────────────────────────────────
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ── Cargar variables de entorno ───────────────────────────────────────────────
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -o allexport
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.env"
    set +o allexport
fi

# ── Ejecutar pipeline ─────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Iniciando pipeline UNAV..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log: ${LOG_FILE}"

python "${SCRIPT_DIR}/src/pipeline.py" "$@" 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE="${PIPESTATUS[0]}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline finalizado con código ${EXIT_CODE}."
exit "${EXIT_CODE}"
