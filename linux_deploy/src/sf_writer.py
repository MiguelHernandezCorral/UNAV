"""
src/sf_writer.py
================
Fase 4 · Write-back de predicciones a Salesforce.

Flujo:
    1. Lee PMAT_PRED_ACTUAL (Oracle) — última predicción por oportunidad
    2. Lee PMAT_SF_SYNC_LOG (Oracle) — último valor enviado por oportunidad
    3. Filtra oportunidades donde PROBABILIDAD ha cambiado
    4. Envía en lotes de 200 via PATCH composite/sobjects
    5. Registra resultado en PMAT_SF_SYNC_LOG

Variables de entorno:
    SF_PROB_FIELD   — campo custom en Opportunity (default: NU_Probabilidad_de_matricula__c)
    SF_URL, SF_CLIENT_ID, SF_CLIENT_SECRET, SF_API_VERSION — igual que sf_extractor
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv()

logger = logging.getLogger(__name__)

SF_PROB_FIELD = os.getenv("SF_PROB_FIELD", "NU_Probabilidad_de_matricula__c")
SF_BATCH_SIZE = 200
SYNC_LOG_TABLE = "PMAT_SF_SYNC_LOG"


# ---------------------------------------------------------------------------
# Autenticación SF (reutiliza SalesforceExtractor)
# ---------------------------------------------------------------------------

def _get_sf_headers() -> tuple[dict, str]:
    """Devuelve (headers con Bearer, instance_url)."""
    from sf_extractor import SalesforceExtractor  # noqa: PLC0415
    sf = SalesforceExtractor()
    sf.authenticate()
    return sf._headers, sf._instance_url


# ---------------------------------------------------------------------------
# Lógica principal
# ---------------------------------------------------------------------------

def _cargar_pred_actual(conn) -> dict[str, float]:
    """Carga {OPP_ID: PROBABILIDAD} desde PMAT_PRED_ACTUAL."""
    rows = conn.read_table("PMAT_PRED_ACTUAL")
    return {r["OPP_ID"]: r["PROBABILIDAD"] for r in rows if r.get("OPP_ID")}


def _cargar_ultimo_envio(conn) -> dict[str, float]:
    """Carga {OPP_ID: PROBABILIDAD_ENV} desde PMAT_SF_SYNC_LOG (último OK)."""
    if not conn._table_exists(SYNC_LOG_TABLE):
        return {}
    rows = conn.read_table(SYNC_LOG_TABLE)
    # Quedar con el registro más reciente por OPP_ID con STATUS='OK'
    ultimos: dict[str, dict] = {}
    for r in rows:
        opp = r.get("OPP_ID")
        if not opp or r.get("STATUS") != "OK":
            continue
        if opp not in ultimos or r["FECHA_ENVIO"] > ultimos[opp]["FECHA_ENVIO"]:
            ultimos[opp] = r
    return {opp: r["PROBABILIDAD_ENV"] for opp, r in ultimos.items()}


def _enviar_lote(headers: dict, instance_url: str, api_version: str,
                 lote: list[dict]) -> list[dict]:
    """Envía un lote de hasta 200 registros via PATCH composite/sobjects."""
    endpoint = f"{instance_url}/services/data/v{api_version}/composite/sobjects"
    payload = {
        "allOrNone": False,
        "records": [
            {
                "attributes": {"type": "Opportunity"},
                "id": r["OPP_ID"],
                SF_PROB_FIELD: round(r["PROBABILIDAD"]),
            }
            for r in lote
        ],
    }
    resp = requests.patch(endpoint, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()  # lista de {id, success, errors}


def _registrar_resultados(conn, lote: list[dict], resultados: list[dict],
                           fecha: datetime) -> None:
    """Inserta filas en PMAT_SF_SYNC_LOG con el resultado de cada envío."""
    log_rows = []
    for rec, res in zip(lote, resultados):
        ok = res.get("success", False)
        errores = res.get("errors", [])
        detalle = str(errores[0]) if errores else ""
        log_rows.append({
            "OPP_ID":          rec["OPP_ID"],
            "PROBABILIDAD_ENV": rec["PROBABILIDAD"],
            "FECHA_ENVIO":     fecha,
            "STATUS":          "OK" if ok else "ERROR",
            "DETALLE":         detalle[:500] if detalle else "",
        })
    conn.insert_records(log_rows, SYNC_LOG_TABLE)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> None:
    """
    Ejecuta el write-back completo a Salesforce.

    Args:
        dry_run: Si True, calcula los cambios pero no llama a SF ni escribe en PMAT_SF_SYNC_LOG.
    """
    from oracle_connector import OracleConnector  # noqa: PLC0415

    conn = OracleConnector()
    fecha = datetime.now()

    logger.info("Cargando predicciones actuales desde PMAT_PRED_ACTUAL...")
    pred_actual = _cargar_pred_actual(conn)
    logger.info("Predicciones cargadas: %d oportunidades.", len(pred_actual))

    logger.info("Cargando últimos valores enviados a SF...")
    ultimo_envio = _cargar_ultimo_envio(conn)
    logger.info("Últimos envíos registrados: %d oportunidades.", len(ultimo_envio))

    # Filtrar oportunidades con cambio en PROBABILIDAD
    a_enviar = [
        {"OPP_ID": opp_id, "PROBABILIDAD": prob}
        for opp_id, prob in pred_actual.items()
        if prob != ultimo_envio.get(opp_id)
    ]
    logger.info("Oportunidades con cambio en probabilidad: %d", len(a_enviar))

    if not a_enviar:
        logger.info("Sin cambios — no se envía nada a Salesforce.")
        return

    if dry_run:
        logger.info("[dry-run] Se enviarían %d oportunidades a SF. Sin escritura.", len(a_enviar))
        return

    headers, instance_url = _get_sf_headers()
    api_version = os.getenv("SF_API_VERSION", "60.0")

    total_ok = 0
    total_err = 0
    for i in range(0, len(a_enviar), SF_BATCH_SIZE):
        lote = a_enviar[i: i + SF_BATCH_SIZE]
        logger.info("Enviando lote %d-%d de %d...",
                    i + 1, i + len(lote), len(a_enviar))
        try:
            resultados = _enviar_lote(headers, instance_url, api_version, lote)
            ok  = sum(1 for r in resultados if r.get("success"))
            err = sum(1 for r in resultados if not r.get("success"))
            total_ok  += ok
            total_err += err
            logger.info("Lote enviado: %d OK, %d errores.", ok, err)
            _registrar_resultados(conn, lote, resultados, fecha)
        except Exception as exc:
            logger.error("Error enviando lote %d-%d: %s", i + 1, i + len(lote), exc)
            # Registrar todo el lote como ERROR
            _registrar_resultados(
                conn, lote,
                [{"success": False, "errors": [str(exc)]}] * len(lote),
                fecha,
            )
            total_err += len(lote)

    logger.info("Write-back SF completado: %d OK | %d errores.", total_ok, total_err)
