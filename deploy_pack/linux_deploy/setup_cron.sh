#!/usr/bin/env bash
# =============================================================================
# setup_cron.sh — Configura la ejecución diaria automática del pipeline UNAV
# =============================================================================
# Añade una entrada en el crontab del usuario actual para lanzar el pipeline
# todos los días a las 03:00 hora del servidor.
#
# Uso:
#   bash setup_cron.sh          # instala el cron
#   bash setup_cron.sh --remove # elimina el cron
#   bash setup_cron.sh --status # muestra el estado actual
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="${SCRIPT_DIR}/run_pipeline.sh"
LOG_FILE="${SCRIPT_DIR}/logs/cron.log"
CRON_LINE="0 3 * * * ${PIPELINE_SCRIPT} >> ${LOG_FILE} 2>&1"
CRON_MARKER="run_pipeline.sh"

# ── Funciones ─────────────────────────────────────────────────────────────────

cron_existe() {
    crontab -l 2>/dev/null | grep -qF "${CRON_MARKER}"
}

mostrar_estado() {
    echo "=== Crontab actual ==="
    crontab -l 2>/dev/null || echo "(vacío)"
    echo "====================="
}

# ── Argumentos ────────────────────────────────────────────────────────────────

ACTION="${1:-install}"

case "${ACTION}" in

    --remove)
        if cron_existe; then
            crontab -l 2>/dev/null | grep -vF "${CRON_MARKER}" | crontab -
            echo "[OK] Cron eliminado."
        else
            echo "[INFO] No había ningún cron configurado para el pipeline."
        fi
        mostrar_estado
        ;;

    --status)
        if cron_existe; then
            echo "[ACTIVO] Pipeline programado:"
            crontab -l 2>/dev/null | grep "${CRON_MARKER}"
        else
            echo "[INACTIVO] No hay cron configurado para el pipeline."
        fi
        mostrar_estado
        ;;

    *)
        # Instalación por defecto
        mkdir -p "${SCRIPT_DIR}/logs"

        if cron_existe; then
            echo "[INFO] El cron ya está configurado:"
            crontab -l 2>/dev/null | grep "${CRON_MARKER}"
            echo ""
            echo "Si quieres reinstalarlo, ejecuta primero: bash setup_cron.sh --remove"
        else
            (crontab -l 2>/dev/null; echo "${CRON_LINE}") | crontab -
            echo "[OK] Cron instalado: ejecución diaria a las 03:00"
            echo "     Línea: ${CRON_LINE}"
            echo ""
            echo "Comandos útiles:"
            echo "  Ver estado:    bash setup_cron.sh --status"
            echo "  Eliminar:      bash setup_cron.sh --remove"
            echo "  Ver log:       tail -f ${LOG_FILE}"
        fi
        mostrar_estado
        ;;
esac
