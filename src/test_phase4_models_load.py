"""
Fase 4 - Tests de carga de modelos preentrenados.

Verifica que los archivos PKL existen en models/ y, si PyCaret está disponible
en el entorno, que cargan correctamente.

Nota: PyCaret requiere Python 3.8-3.11. En entornos con Python 3.12+
los tests de carga se marcan como SKIP automáticamente.
"""
import pytest
from pathlib import Path

try:
    from pycaret.classification import load_model as pycaret_load_model
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False

# Raíz del proyecto relativa a este archivo (src/ -> ../)
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

pycaret_required = pytest.mark.skipif(
    not PYCARET_AVAILABLE,
    reason="PyCaret no está instalado en este entorno (requiere Python 3.8-3.11)"
)


# ─── Tests de existencia de archivos (siempre ejecutan) ───────────────────────

def test_models_dir_exists():
    """La carpeta models/ debe existir en la raíz del proyecto."""
    assert MODELS_DIR.is_dir(), f"No existe la carpeta {MODELS_DIR}"


def test_grado_pkl_exists():
    """El archivo modelo_final_grado.pkl debe estar en models/."""
    pkl = MODELS_DIR / "modelo_final_grado.pkl"
    assert pkl.is_file(), f"No se encontró {pkl}"


def test_master_pkl_exists():
    """El archivo modelo_final_master.pkl debe estar en models/."""
    pkl = MODELS_DIR / "modelo_final_master.pkl"
    assert pkl.is_file(), f"No se encontró {pkl}"


def test_grado_pkl_not_empty():
    """El PKL de Grado no debe estar vacío."""
    pkl = MODELS_DIR / "modelo_final_grado.pkl"
    assert pkl.stat().st_size > 0, "modelo_final_grado.pkl está vacío"


def test_master_pkl_not_empty():
    """El PKL de Máster no debe estar vacío."""
    pkl = MODELS_DIR / "modelo_final_master.pkl"
    assert pkl.stat().st_size > 0, "modelo_final_master.pkl está vacío"


# ─── Tests de carga con PyCaret (skip si no disponible) ───────────────────────

@pycaret_required
def test_grado_model_loads():
    """El modelo de Grado debe cargar sin errores con PyCaret load_model."""
    ruta = str(MODELS_DIR / "modelo_final_grado")
    model = pycaret_load_model(ruta, verbose=False)
    assert model is not None, "load_model devolvió None para modelo_grado"


@pycaret_required
def test_master_model_loads():
    """El modelo de Máster debe cargar sin errores con PyCaret load_model."""
    ruta = str(MODELS_DIR / "modelo_final_master")
    model = pycaret_load_model(ruta, verbose=False)
    assert model is not None, "load_model devolvió None para modelo_master"


@pycaret_required
def test_models_have_predict_method():
    """Los modelos cargados deben tener atributo 'predict' (pipeline sklearn)."""
    for nombre in ["modelo_final_grado", "modelo_final_master"]:
        model = pycaret_load_model(str(MODELS_DIR / nombre), verbose=False)
        assert hasattr(model, "predict"), (
            f"{nombre} no tiene método predict — puede no ser un pipeline sklearn válido"
        )
