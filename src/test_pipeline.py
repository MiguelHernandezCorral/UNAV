"""
Fase B - Tests del orquestador pipeline.py

Tests sin conexión Oracle ni Salesforce: usan dry_run y mocks de fase.
"""
import sys
import logging
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR      = Path(__file__).parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline import run_pipeline, PHASE_REGISTRY, PHASE_ORDER, _run_phase


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mock_module(phase_name: str, side_effect=None):
    """Devuelve un context manager que mockea el módulo de una fase."""
    cfg     = PHASE_REGISTRY[phase_name]
    mod_key = f"{cfg['module']}.{cfg['entry']}"
    mock_fn = MagicMock(side_effect=side_effect)
    return patch(mod_key, mock_fn), mock_fn


# ─── Tests: run_pipeline dry-run ──────────────────────────────────────────────

def test_dry_run_all_phases_retorna_ok():
    """dry_run=True ejecuta todas las fases con status 'ok' sin Oracle/SF."""
    resultado = run_pipeline(phases=["all"], dry_run=True)
    assert set(resultado.keys()) == set(PHASE_ORDER)
    for fase, info in resultado.items():
        assert info["status"] == "ok", f"Fase {fase} no es 'ok': {info}"


def test_dry_run_fase_unica():
    """dry_run con una sola fase devuelve solo esa fase."""
    resultado = run_pipeline(phases=["fase4"], dry_run=True)
    assert "fase4" in resultado
    assert resultado["fase4"]["status"] == "ok"
    # Las demás fases no deben estar en el resultado
    assert "fase1" not in resultado
    assert "fase2" not in resultado


def test_dry_run_fases_multiples():
    resultado = run_pipeline(phases=["fase1", "fase2"], dry_run=True)
    assert set(resultado.keys()) == {"fase1", "fase2"}
    for info in resultado.values():
        assert info["status"] == "ok"


# ─── Tests: stop_on_error ────────────────────────────────────────────────────

def test_stop_on_error_para_en_primer_fallo():
    """Con stop_on_error=True, tras fallo de fase1 no se ejecutan fase2 ni fase4."""
    llamadas = []

    def fase1_falla(**kw):
        llamadas.append("fase1")
        raise RuntimeError("SF no disponible")

    def fase2_ok(**kw):
        llamadas.append("fase2")

    with patch("sf_extract_all.run", side_effect=fase1_falla), \
         patch("cleaner.run",        side_effect=fase2_ok):
        resultado = run_pipeline(
            phases=["fase1", "fase2"],
            dry_run=False,
            stop_on_error=True,
        )

    assert resultado["fase1"]["status"] == "error"
    assert "fase2" not in resultado
    assert "fase2" not in llamadas


def test_continue_on_error_ejecuta_fase2_tras_fallo_fase1():
    """Con stop_on_error=False, fase2 se ejecuta aunque fase1 falle."""
    llamadas = []

    def fase1_falla(**kw):
        llamadas.append("fase1")
        raise RuntimeError("SF no disponible")

    def fase2_ok(**kw):
        llamadas.append("fase2")

    with patch("sf_extract_all.run", side_effect=fase1_falla), \
         patch("cleaner.run",        side_effect=fase2_ok):
        resultado = run_pipeline(
            phases=["fase1", "fase2"],
            dry_run=False,
            stop_on_error=False,
        )

    assert resultado["fase1"]["status"] == "error"
    assert resultado["fase2"]["status"] == "ok"
    assert "fase2" in llamadas


# ─── Tests: estructura del resultado ─────────────────────────────────────────

def test_resultado_tiene_status_y_duration():
    """El resultado de cada fase tiene 'status' y 'duration_s'."""
    resultado = run_pipeline(phases=["fase4"], dry_run=True)
    for info in resultado.values():
        assert "status"     in info
        assert "duration_s" in info
        assert isinstance(info["duration_s"], float)


def test_resultado_error_tiene_campo_error():
    def fase2_falla(**kw):
        raise ValueError("Error de limpieza")

    with patch("cleaner.run", side_effect=fase2_falla):
        resultado = run_pipeline(
            phases=["fase2"],
            dry_run=False,
            stop_on_error=False,
        )

    assert resultado["fase2"]["status"] == "error"
    assert "error" in resultado["fase2"]
    assert "Error de limpieza" in resultado["fase2"]["error"]


# ─── Tests: fase desconocida ──────────────────────────────────────────────────

def test_fase_desconocida_lanza_valueerror():
    with pytest.raises(ValueError, match="desconocidas"):
        run_pipeline(phases=["fase99"])


def test_fase_desconocida_en_lista_mixta():
    with pytest.raises(ValueError):
        run_pipeline(phases=["fase1", "fase_inventada"])


# ─── Tests: orden de ejecución ───────────────────────────────────────────────

def test_orden_ejecucion_respeta_phase_order():
    """Las fases se ejecutan en el orden de PHASE_ORDER, no en el de la lista."""
    orden_real = []

    def fase1_ok(**kw): orden_real.append("fase1")
    def fase2_ok(**kw): orden_real.append("fase2")

    with patch("sf_extract_all.run", side_effect=fase1_ok), \
         patch("cleaner.run",        side_effect=fase2_ok):
        run_pipeline(phases=["fase2", "fase1"], dry_run=False)

    assert orden_real == ["fase1", "fase2"]


# ─── Tests: log file ─────────────────────────────────────────────────────────

def test_log_dir_creado_tras_ejecucion(tmp_path, monkeypatch):
    """El directorio de logs se crea automáticamente."""
    import pipeline as pipeline_mod
    fake_logs = tmp_path / "logs"
    monkeypatch.setattr(pipeline_mod, "LOGS_DIR", fake_logs)
    run_pipeline(phases=["fase4"], dry_run=True)
    assert fake_logs.is_dir()


# ─── Tests: _run_phase directo ───────────────────────────────────────────────

def test_run_phase_dry_run_no_importa_modulo():
    """_run_phase con dry_run=True no debe importar ni ejecutar el módulo."""
    with patch("importlib.import_module") as mock_import:
        _run_phase("fase1", dry_run=True)
    mock_import.assert_not_called()


def test_run_phase_fase_desconocida_lanza_error():
    with pytest.raises(ValueError):
        _run_phase("fase_invalida")
