"""
Tests unitarios para src/sf_writer.py

Grupos:
    1. _cargar_pred_actual     — parseo de filas de PMAT_PRED_ACTUAL
    2. _cargar_ultimo_envio    — lógica de "último OK por OPP_ID"
    3. _enviar_lote            — payload PATCH composite/sobjects
    4. _registrar_resultados   — filas escritas en PMAT_SF_SYNC_LOG
    5. run()                   — orquestación completa (mock Oracle + requests)

Sin conexión Oracle ni Salesforce real — todo mockeado.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sf_writer import (
    _cargar_pred_actual,
    _cargar_ultimo_envio,
    _enviar_lote,
    _registrar_resultados,
    SF_PROB_FIELD,
    SF_CONF_FIELD,
    SYNC_LOG_TABLE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rows_pred(n: int = 3) -> list[dict]:
    return [
        {"OPP_ID": f"OPP{i:03d}", "PROBABILIDAD": float(10 * i + 40), "CONFIANZA": float(5 * i + 60)}
        for i in range(n)
    ]


def _make_rows_log(n: int = 3, status: str = "OK") -> list[dict]:
    return [
        {
            "OPP_ID":           f"OPP{i:03d}",
            "PROBABILIDAD_ENV": float(10 * i + 40),
            "CONFIANZA_ENV":    float(5 * i + 60),
            "FECHA_ENVIO":      datetime(2026, 3, 25, 3, 0, 0),
            "STATUS":           status,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. _cargar_pred_actual
# ---------------------------------------------------------------------------

def test_cargar_pred_actual_devuelve_dict():
    conn = MagicMock()
    conn.read_table.return_value = _make_rows_pred(3)
    result = _cargar_pred_actual(conn)
    assert isinstance(result, dict)
    assert len(result) == 3


def test_cargar_pred_actual_claves_opp_id():
    conn = MagicMock()
    conn.read_table.return_value = _make_rows_pred(3)
    result = _cargar_pred_actual(conn)
    assert set(result.keys()) == {"OPP000", "OPP001", "OPP002"}


def test_cargar_pred_actual_contiene_prob_y_confianza():
    conn = MagicMock()
    conn.read_table.return_value = _make_rows_pred(1)
    result = _cargar_pred_actual(conn)
    datos = result["OPP000"]
    assert "PROBABILIDAD" in datos
    assert "CONFIANZA" in datos


def test_cargar_pred_actual_ignora_filas_sin_opp_id():
    conn = MagicMock()
    conn.read_table.return_value = [
        {"OPP_ID": "OPP001", "PROBABILIDAD": 55.0, "CONFIANZA": 70.0},
        {"OPP_ID": None,     "PROBABILIDAD": 30.0, "CONFIANZA": 40.0},
        {"OPP_ID": "",       "PROBABILIDAD": 20.0, "CONFIANZA": 30.0},
    ]
    result = _cargar_pred_actual(conn)
    assert list(result.keys()) == ["OPP001"]


def test_cargar_pred_actual_confianza_none_se_trata_como_cero():
    conn = MagicMock()
    conn.read_table.return_value = [
        {"OPP_ID": "OPP001", "PROBABILIDAD": 55.0, "CONFIANZA": None},
    ]
    result = _cargar_pred_actual(conn)
    assert result["OPP001"]["CONFIANZA"] == 0


# ---------------------------------------------------------------------------
# 2. _cargar_ultimo_envio
# ---------------------------------------------------------------------------

def test_cargar_ultimo_envio_tabla_inexistente_devuelve_vacio():
    conn = MagicMock()
    conn._table_exists.return_value = False
    result = _cargar_ultimo_envio(conn)
    assert result == {}


def test_cargar_ultimo_envio_solo_ok():
    conn = MagicMock()
    conn._table_exists.return_value = True
    rows = _make_rows_log(2, status="OK") + _make_rows_log(1, status="ERROR")
    conn.read_table.return_value = rows
    result = _cargar_ultimo_envio(conn)
    # Solo los OK deben estar en el resultado
    assert all(k.startswith("OPP") for k in result)
    # Los ERROR no deben aparecer
    for key in result:
        matching = [r for r in rows if r["OPP_ID"] == key]
        assert any(r["STATUS"] == "OK" for r in matching)


def test_cargar_ultimo_envio_queda_con_mas_reciente():
    conn = MagicMock()
    conn._table_exists.return_value = True
    conn.read_table.return_value = [
        {"OPP_ID": "OPP001", "PROBABILIDAD_ENV": 40.0, "FECHA_ENVIO": datetime(2026, 3, 24), "STATUS": "OK"},
        {"OPP_ID": "OPP001", "PROBABILIDAD_ENV": 55.0, "FECHA_ENVIO": datetime(2026, 3, 25), "STATUS": "OK"},
    ]
    result = _cargar_ultimo_envio(conn)
    assert result["OPP001"] == 55.0


# ---------------------------------------------------------------------------
# 3. _enviar_lote
# ---------------------------------------------------------------------------

def _lote_ejemplo(n: int = 2) -> list[dict]:
    return [
        {"OPP_ID": f"OPP{i:03d}", "PROBABILIDAD": float(10 * i + 40), "CONFIANZA": float(5 * i + 60)}
        for i in range(n)
    ]


def test_enviar_lote_payload_all_or_none_false():
    lote = _lote_ejemplo(2)
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"id": r["OPP_ID"], "success": True, "errors": []} for r in lote]

    with patch("requests.patch", return_value=mock_resp) as mock_patch:
        _enviar_lote({}, "https://sf.example.com", "60.0", lote)

    payload_enviado = mock_patch.call_args[1]["json"]
    assert payload_enviado["allOrNone"] is False


def test_enviar_lote_contiene_campo_probabilidad():
    lote = _lote_ejemplo(2)
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"id": r["OPP_ID"], "success": True, "errors": []} for r in lote]

    with patch("requests.patch", return_value=mock_resp) as mock_patch:
        _enviar_lote({}, "https://sf.example.com", "60.0", lote)

    records = mock_patch.call_args[1]["json"]["records"]
    for rec in records:
        assert SF_PROB_FIELD in rec


def test_enviar_lote_contiene_campo_confianza():
    lote = _lote_ejemplo(2)
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"id": r["OPP_ID"], "success": True, "errors": []} for r in lote]

    with patch("requests.patch", return_value=mock_resp) as mock_patch:
        _enviar_lote({}, "https://sf.example.com", "60.0", lote)

    records = mock_patch.call_args[1]["json"]["records"]
    for rec in records:
        assert SF_CONF_FIELD in rec


def test_enviar_lote_probabilidad_redondeada():
    lote = [{"OPP_ID": "OPP001", "PROBABILIDAD": 67.89, "CONFIANZA": 55.4}]
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"id": "OPP001", "success": True, "errors": []}]

    with patch("requests.patch", return_value=mock_resp) as mock_patch:
        _enviar_lote({}, "https://sf.example.com", "60.0", lote)

    record = mock_patch.call_args[1]["json"]["records"][0]
    assert record[SF_PROB_FIELD] == 68       # round(67.89)
    assert record[SF_CONF_FIELD] == 55       # round(55.4)
    assert isinstance(record[SF_PROB_FIELD], int)
    assert isinstance(record[SF_CONF_FIELD], int)


def test_enviar_lote_endpoint_correcto():
    lote = _lote_ejemplo(1)
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"id": "OPP000", "success": True, "errors": []}]

    with patch("requests.patch", return_value=mock_resp) as mock_patch:
        _enviar_lote({}, "https://sf.example.com", "60.0", lote)

    url_llamada = mock_patch.call_args[0][0]
    assert url_llamada == "https://sf.example.com/services/data/v60.0/composite/sobjects"


def test_enviar_lote_devuelve_lista_resultados():
    lote = _lote_ejemplo(3)
    esperado = [{"id": r["OPP_ID"], "success": True, "errors": []} for r in lote]
    mock_resp = MagicMock()
    mock_resp.json.return_value = esperado

    with patch("requests.patch", return_value=mock_resp):
        result = _enviar_lote({}, "https://sf.example.com", "60.0", lote)

    assert result == esperado


# ---------------------------------------------------------------------------
# 4. _registrar_resultados
# ---------------------------------------------------------------------------

def test_registrar_resultados_inserta_en_sync_log():
    conn = MagicMock()
    lote = _lote_ejemplo(2)
    resultados = [{"success": True, "errors": []}, {"success": False, "errors": [{"message": "ERR"}]}]
    _registrar_resultados(conn, lote, resultados, datetime.now())
    conn.insert_records.assert_called_once()


def test_registrar_resultados_status_ok_y_error():
    conn = MagicMock()
    lote = _lote_ejemplo(2)
    resultados = [{"success": True, "errors": []}, {"success": False, "errors": [{"message": "Campo requerido"}]}]
    _registrar_resultados(conn, lote, resultados, datetime.now())

    filas = conn.insert_records.call_args[0][0]
    assert filas[0]["STATUS"] == "OK"
    assert filas[1]["STATUS"] == "ERROR"


def test_registrar_resultados_incluye_confianza_env():
    conn = MagicMock()
    lote = [{"OPP_ID": "OPP001", "PROBABILIDAD": 65.0, "CONFIANZA": 72.0}]
    resultados = [{"success": True, "errors": []}]
    _registrar_resultados(conn, lote, resultados, datetime.now())

    fila = conn.insert_records.call_args[0][0][0]
    assert "CONFIANZA_ENV" in fila
    assert fila["CONFIANZA_ENV"] == 72.0


def test_registrar_resultados_detalle_truncado():
    conn = MagicMock()
    lote = [{"OPP_ID": "OPP001", "PROBABILIDAD": 50.0, "CONFIANZA": 60.0}]
    msg_largo = "X" * 600
    resultados = [{"success": False, "errors": [msg_largo]}]
    _registrar_resultados(conn, lote, resultados, datetime.now())

    fila = conn.insert_records.call_args[0][0][0]
    assert len(fila["DETALLE"]) <= 500


# ---------------------------------------------------------------------------
# 5. run() — orquestación completa
# ---------------------------------------------------------------------------

def _mock_conn(pred_rows=None, log_rows=None, tabla_log_existe=True):
    """Construye un OracleConnector mockeado con datos configurables."""
    conn = MagicMock()
    conn._table_exists.return_value = tabla_log_existe

    pred_rows = pred_rows or _make_rows_pred(3)
    log_rows  = log_rows  or []

    def read_table_side(tabla):
        if tabla == "PMAT_PRED_ACTUAL":
            return pred_rows
        if tabla == SYNC_LOG_TABLE:
            return log_rows
        return []

    conn.read_table.side_effect = read_table_side
    return conn


def test_run_dry_run_no_llama_requests():
    conn = _mock_conn()
    with patch("oracle_connector.OracleConnector", return_value=conn), \
         patch("requests.patch") as mock_req:
        from sf_writer import run
        run(dry_run=True)
    mock_req.assert_not_called()


def test_run_dry_run_no_escribe_sync_log():
    conn = _mock_conn()
    with patch("oracle_connector.OracleConnector", return_value=conn):
        from sf_writer import run
        run(dry_run=True)
    conn.insert_records.assert_not_called()


def test_run_sin_cambios_no_llama_sf():
    """Si PROBABILIDAD no cambió, no debe enviarse nada."""
    pred_rows = _make_rows_pred(2)
    # El log ya tiene exactamente los mismos valores
    log_rows = [
        {
            "OPP_ID": r["OPP_ID"],
            "PROBABILIDAD_ENV": r["PROBABILIDAD"],
            "CONFIANZA_ENV":    r["CONFIANZA"],
            "FECHA_ENVIO":      datetime(2026, 3, 25),
            "STATUS":           "OK",
        }
        for r in pred_rows
    ]
    conn = _mock_conn(pred_rows=pred_rows, log_rows=log_rows)

    with patch("oracle_connector.OracleConnector", return_value=conn), \
         patch("requests.patch") as mock_req:
        from sf_writer import run
        run(dry_run=False)

    mock_req.assert_not_called()


def test_run_con_cambios_llama_sf():
    """Si hay oportunidades con PROBABILIDAD cambiada, debe llamar a SF."""
    pred_rows = _make_rows_pred(2)
    log_rows  = []  # primer envío: no hay historial
    conn = _mock_conn(pred_rows=pred_rows, log_rows=log_rows)

    mock_resp = MagicMock()
    mock_resp.json.return_value = [
        {"id": r["OPP_ID"], "success": True, "errors": []} for r in pred_rows
    ]

    with patch("oracle_connector.OracleConnector", return_value=conn), \
         patch("sf_writer._get_sf_headers", return_value=({}, "https://sf.example.com")), \
         patch("requests.patch", return_value=mock_resp) as mock_req:
        from sf_writer import run
        run(dry_run=False)

    mock_req.assert_called()


def test_run_migra_columna_confianza_env():
    """run() debe llamar a add_column_if_not_exists para CONFIANZA_ENV."""
    conn = _mock_conn()
    mock_resp = MagicMock()
    mock_resp.json.return_value = [
        {"id": r["OPP_ID"], "success": True, "errors": []} for r in _make_rows_pred(3)
    ]

    with patch("oracle_connector.OracleConnector", return_value=conn), \
         patch("sf_writer._get_sf_headers", return_value=({}, "https://sf.example.com")), \
         patch("requests.patch", return_value=mock_resp):
        from sf_writer import run
        run(dry_run=False)

    conn.add_column_if_not_exists.assert_called_with(SYNC_LOG_TABLE, "CONFIANZA_ENV", "FLOAT")
