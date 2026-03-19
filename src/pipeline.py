"""
src/pipeline.py
===============
Orquestador del pipeline completo UNAV.

Ejecuta las fases del pipeline en secuencia:
    fase1    — Ingesta Salesforce → Oracle  (sf_extract_all.run)
    fase2    — Limpieza → DATASET_LIMPIO    (cleaner.run)
    validate — Validación pre-predicciones  (validator.run_validation_phase)
    fase4    — Predicciones → PREDICCIONES_V2 (predictor.run_predictions_v2)

Uso directo:
    python src/pipeline.py                         # todas las fases
    python src/pipeline.py --phases fase2 fase4    # fases concretas
    python src/pipeline.py --dry-run               # sin escribir en Oracle/SF
    python src/pipeline.py --no-stop-on-error      # continúa aunque falle una fase
    python src/pipeline.py --phases fase4 --save-hist  # guarda historial de predicciones

Desde bash (Linux/Mac):
    bash run_pipeline.sh [opciones]
"""

import importlib
import logging
import sys
import time
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

# ─── Paths ────────────────────────────────────────────────────────────────────

SRC_DIR      = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
LOGS_DIR     = PROJECT_ROOT / "logs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ─── Logging ──────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


def _setup_logging(level: int = logging.INFO) -> None:
    """Configura logging a consola y fichero rotatorio."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "pipeline.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    if root.handlers:
        return  # ya configurado por un módulo importado

    fh = TimedRotatingFileHandler(
        log_file, when="midnight", backupCount=30, encoding="utf-8"
    )
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    root.setLevel(level)
    root.addHandler(fh)
    root.addHandler(sh)


# ─── Registry de fases ────────────────────────────────────────────────────────

PHASE_REGISTRY: dict[str, dict[str, Any]] = {
    "fase1": {
        "name":   "Ingesta Salesforce → Oracle",
        "module": "sf_extract_all",
        "entry":  "run",
        "kwargs": {},
    },
    "fase2": {
        "name":   "Limpieza → DATASET_LIMPIO",
        "module": "cleaner",
        "entry":  "run",
        "kwargs": {},
    },
    "validate": {
        "name":   "Validación pre-predicciones",
        "module": "validator",
        "entry":  "run_validation_phase",
        "kwargs": {},
    },
    "fase4": {
        "name":   "Predicciones → PREDICCIONES_V2",
        "module": "predictor",
        "entry":  "run_predictions_v2",
        "kwargs": {"save_to_oracle": True, "return_df": False},
    },
}

PHASE_ORDER = ["fase1", "fase2", "validate", "fase4"]


# ─── Ejecución de una fase ────────────────────────────────────────────────────

def _run_phase(
    phase_name: str,
    dry_run: bool = False,
    extra_kwargs: dict | None = None,
) -> None:
    """
    Importa y ejecuta una fase del pipeline.

    Args:
        phase_name:   Clave en PHASE_REGISTRY.
        dry_run:      Si True, loguea la acción sin ejecutarla.
        extra_kwargs: Kwargs adicionales a mezclar con los del registro.

    Raises:
        ValueError: Si phase_name no está en PHASE_REGISTRY.
        Exception:  Cualquier error propagado desde la fase.
    """
    if phase_name not in PHASE_REGISTRY:
        raise ValueError(
            f"Fase desconocida: '{phase_name}'. "
            f"Válidas: {list(PHASE_REGISTRY.keys())}"
        )

    cfg = PHASE_REGISTRY[phase_name]
    name = cfg["name"]

    if dry_run:
        logger.info("[DRY-RUN] Fase '%s' (%s) — omitida.", phase_name, name)
        return

    logger.info("▶ Iniciando fase '%s': %s", phase_name, name)
    mod = importlib.import_module(cfg["module"])
    fn  = getattr(mod, cfg["entry"])

    kwargs = dict(cfg.get("kwargs", {}))
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    fn(**kwargs)
    logger.info("✓ Fase '%s' completada.", phase_name)


# ─── Función principal ────────────────────────────────────────────────────────

def run_pipeline(
    phases: list[str] | None = None,
    dry_run: bool = False,
    stop_on_error: bool = True,
    save_hist: bool = False,
) -> dict[str, Any]:
    """
    Ejecuta las fases del pipeline en secuencia.

    Args:
        phases:        Lista de fases a ejecutar, o None / ['all'] para todas.
        dry_run:       Si True, no escribe datos en Oracle ni Salesforce.
        stop_on_error: Si True, para en el primer fallo.
        save_hist:     Si True, guarda también en PREDICCIONES_V2_HIST (solo fase4).

    Returns:
        dict {phase_name: {status: 'ok'|'error', duration_s: float, error?: str}}
    """
    phases = phases or ["all"]

    if phases == ["all"] or phases == "all":
        fases_a_ejecutar = PHASE_ORDER
    else:
        desconocidas = [p for p in phases if p not in PHASE_REGISTRY]
        if desconocidas:
            raise ValueError(
                f"Fases desconocidas: {desconocidas}. "
                f"Válidas: {list(PHASE_REGISTRY.keys())}"
            )
        fases_a_ejecutar = [p for p in PHASE_ORDER if p in phases]

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    inicio_total = datetime.now()
    logger.info("=" * 60)
    logger.info("PIPELINE UNAV — %s", inicio_total.isoformat())
    logger.info("Fases: %s | dry_run=%s | stop_on_error=%s",
                fases_a_ejecutar, dry_run, stop_on_error)
    logger.info("=" * 60)

    resultados: dict[str, Any] = {}

    for phase_name in fases_a_ejecutar:
        t0 = time.monotonic()
        extra = {}
        if phase_name == "fase4":
            extra = {"save_hist": save_hist, "dry_run": dry_run}

        try:
            _run_phase(phase_name, dry_run=(dry_run and phase_name != "fase4"), extra_kwargs=extra)
            resultados[phase_name] = {
                "status":     "ok",
                "duration_s": round(time.monotonic() - t0, 2),
            }
        except Exception as exc:
            dur = round(time.monotonic() - t0, 2)
            logger.error(
                "✗ Fase '%s' falló tras %.1f s: %s",
                phase_name, dur, exc,
                exc_info=True,
            )
            resultados[phase_name] = {
                "status":     "error",
                "duration_s": dur,
                "error":      str(exc),
            }
            if stop_on_error:
                logger.error("Pipeline detenido por error en '%s'.", phase_name)
                break

    dur_total = (datetime.now() - inicio_total).total_seconds()
    ok_count  = sum(1 for r in resultados.values() if r["status"] == "ok")
    err_count = sum(1 for r in resultados.values() if r["status"] == "error")

    logger.info("=" * 60)
    logger.info(
        "PIPELINE finalizado en %.1f s | OK: %d | Errores: %d",
        dur_total, ok_count, err_count,
    )
    logger.info("=" * 60)

    return resultados


# ─── Entry point ─────────────────────────────────────────────────────────────

def main(args=None):
    """Entry point CLI y programático."""
    import argparse

    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Pipeline orquestador UNAV: SF → Oracle → Limpieza → Predicciones"
    )
    parser.add_argument(
        "--phases", nargs="+", default=["all"],
        metavar="FASE",
        help="Fases a ejecutar: fase1 fase2 validate fase4 (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Ejecuta el pipeline sin escribir datos en Oracle ni Salesforce",
    )
    parser.add_argument(
        "--no-stop-on-error", action="store_true",
        help="Continúa aunque una fase falle",
    )
    parser.add_argument(
        "--save-hist", action="store_true",
        help="Guarda también en PREDICCIONES_V2_HIST (solo afecta a fase4)",
    )

    parsed = parser.parse_args(args)

    resultados = run_pipeline(
        phases=parsed.phases,
        dry_run=parsed.dry_run,
        stop_on_error=not parsed.no_stop_on_error,
        save_hist=parsed.save_hist,
    )

    # Código de salida: 1 si alguna fase falló
    n_errores = sum(1 for r in resultados.values() if r["status"] == "error")
    sys.exit(n_errores > 0)


if __name__ == "__main__":
    main()
