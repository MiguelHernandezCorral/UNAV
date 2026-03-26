"""
Fase B - Tests del orquestador pipeline.py

Tests sin conexión Oracle ni Salesforce: usan dry_run y mocks de fase.

--- NOTAS DE REVISIÓN (Juan, 26-mar-2026) ---

La estructura de estos tests es muy buena: el helper _mock_module es reutilizable
y el uso de dry_run para los tests de estructura es limpio. Algunos puntos a revisar:

(1) test_dry_run_fase_unica y otros tests usan phases=["fase4"] con dry_run=True.
    Con el comportamiento actual del pipeline, fase4 (sf_writer) NO se salta aunque
    dry_run=True: el pipeline le pasa dry_run como kwarg a sf_writer.run, que lo
    gestiona internamente. Esto significa que el pipeline intentará importar
    oracle_connector y conectarse a Oracle aunque dry_run=True.
    En local (sin Oracle) estos tests pueden fallar. La opción más limpia es
    mockear el módulo sf_writer completo para estos tests de estructura. Ver TAREA 1.

(2) test_stop_on_error_para_en_primer_fallo: el comentario dice "ni fase4" pero
    el test solo usa phases=["fase1", "fase2"], así que fase4 nunca entra.
    Actualizar el docstring para que sea preciso y no confunda.

(3) TAREA PROGRESIVA para Mario (ver al final del fichero).
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
    # REVISIÓN (Juan): con dry_run=True, fase4 (sf_writer) no se omite por completo
    # — el pipeline le pasa dry_run=True como kwarg a sf_writer.run, que lo gestiona
    # internamente, pero sí intenta importar el módulo y conectarse a Oracle.
    # En CI/local sin Oracle, este test puede fallar. Una solución robusta es:
    #   with patch("sf_writer.run") as mock_sf:
    #       resultado = run_pipeline(phases=["fase4"], dry_run=True)
    #   mock_sf.assert_called_once_with(dry_run=True)
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
    """Con stop_on_error=True, tras fallo de fase1 no se ejecuta fase2."""
    # REVISIÓN (Juan): el docstring mencionaba "ni fase4" pero este test solo
    # usa phases=["fase1", "fase2"], así que fase4 nunca se evaluaría de todas formas.
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


# ─── TAREAS PROGRESIVAS PARA MARIO ────────────────────────────────────────────
#
# TAREA 1 (fácil) — Hacer test_dry_run_fase_unica robusto sin Oracle:
#   Modificar el test para que mockee sf_writer.run antes de llamar a run_pipeline.
#   Así el test no necesita Oracle y es más rápido.
#   Pista: usar `with patch("sf_writer.run") as mock_sf:` dentro del test.
#
# TAREA 2 (intermedia) — Test de resultado de fase4:
#   Escribe test_fase4_dry_run_llama_sf_writer_con_dry_run que verifique que
#   cuando se ejecuta run_pipeline(phases=["fase4"], dry_run=True),
#   sf_writer.run se llama con dry_run=True.
#   Pista: patch "sf_writer.run" y luego inspect call_args.
#
# TAREA 3 (intermedia) — Test de orden completo fase1→fase4:
#   Escribe un test que mockee los 4 módulos (sf_extract_all.run, cleaner.run,
#   predictor.run_predictions_v2, sf_writer.run) y verifique que se ejecutan
#   en ese orden cuando se llama a run_pipeline(phases=["all"]).
#   Pista: usa una lista `orden_real = []` y appends en cada mock.
#
# TAREA 4 (reto) — Cobertura de pytest con --cov:
#   Instala pytest-cov (`pip install pytest-cov`) y ejecuta:
#     pytest src/ --cov=src --cov-report=term-missing -v
#   Revisa qué funciones del pipeline no tienen cobertura y añade tests
#   para las más importantes.
# ──────────────────────────────────────────────────────────────────────────────
