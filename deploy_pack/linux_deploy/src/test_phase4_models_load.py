"""
Fase 4 - Tests de carga de modelos preentrenados.

Tests 100% reales: cargan los PKL con PyCaret y verifican su estructura.
Requieren Python 3.8-3.11 con PyCaret instalado.
"""
import pytest
from pathlib import Path

from pycaret.classification import load_model

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


# ─── Tests de existencia de archivos ──────────────────────────────────────────

def test_models_dir_exists():
    """La carpeta models/ debe existir en la raíz del proyecto."""
    assert MODELS_DIR.is_dir(), f"No existe la carpeta {MODELS_DIR}"


def test_grado_pkl_exists():
    """El archivo modelo_final_grado.pkl debe estar en models/."""
    assert (MODELS_DIR / "modelo_final_grado.pkl").is_file()


def test_master_pkl_exists():
    """El archivo modelo_final_master.pkl debe estar en models/."""
    assert (MODELS_DIR / "modelo_final_master.pkl").is_file()


def test_grado_pkl_not_empty():
    assert (MODELS_DIR / "modelo_final_grado.pkl").stat().st_size > 0


def test_master_pkl_not_empty():
    assert (MODELS_DIR / "modelo_final_master.pkl").stat().st_size > 0


# ─── Tests de carga real con PyCaret ──────────────────────────────────────────

def test_grado_model_loads():
    """El modelo de Grado debe cargar sin errores con PyCaret."""
    model = load_model(str(MODELS_DIR / "modelo_final_grado"), verbose=False)
    assert model is not None


def test_master_model_loads():
    """El modelo de Máster debe cargar sin errores con PyCaret."""
    model = load_model(str(MODELS_DIR / "modelo_final_master"), verbose=False)
    assert model is not None


def test_models_have_predict_method():
    """Los modelos cargados deben tener método predict (pipeline sklearn)."""
    for nombre in ["modelo_final_grado", "modelo_final_master"]:
        model = load_model(str(MODELS_DIR / nombre), verbose=False)
        assert hasattr(model, "predict"), f"{nombre} no tiene método predict"


def test_grado_model_has_expected_features():
    """El modelo de Grado debe tener las features esperadas."""
    model = load_model(str(MODELS_DIR / "modelo_final_grado"), verbose=False)
    features = list(model.feature_names_in_)
    expected = [
        "NU_NOTA_MEDIA_ADMISION", "NU_NOTA_MEDIA_1_BACH__PC",
        "etapa_ordinal_num", "vinculacion_previa",
    ]
    for feat in expected:
        assert feat in features, f"Feature '{feat}' no encontrada en modelo_grado"


def test_master_model_has_expected_features():
    """El modelo de Máster debe tener las features esperadas."""
    model = load_model(str(MODELS_DIR / "modelo_final_master"), verbose=False)
    features = list(model.feature_names_in_)
    expected = [
        "NU_NOTA_MEDIA_1_BACH__PC", "vinculacion_previa",
        "etapa_ordinal_num", "CU_IMPORTE_TOTAL",
    ]
    for feat in expected:
        assert feat in features, f"Feature '{feat}' no encontrada en modelo_master"
