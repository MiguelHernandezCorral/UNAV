"""
Fase 4 - Tests reales del predictor con PyCaret.

Tests 100% reales: usan los modelos PKL reales y DataFrames sintéticos
con las columnas exactas que espera cada modelo. Sin mocks.

Requieren Python 3.8-3.11 con PyCaret instalado.
"""
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pycaret.classification import load_model, predict_model
from predictor import predecir, construir_resultado, cargar_modelo, run_predictions

MODELS_DIR = PROJECT_ROOT / "models"

# ─── Columnas exactas que esperan los modelos ─────────────────────────────────

FEATURES_GRADO = [
    "NU_NOTA_MEDIA_ADMISION", "NU_NOTA_MEDIA_1_BACH__PC",
    "NU_RESULTADO_ADMISION_PUNTOS", "PAID_PERCENT",
    "CH_MATRICULA_SUJETA_BECA", "CU_IMPORTE_TOTAL",
    "NU_PREFERENCIA", "PL_SITUACION_SOCIO_ECONOMICA",
    "tiempo_etapa_dias", "tiempo_entre_etapas_dias",
    "num_asistencias_acum", "num_solicitudes_acum",
    "NU_MEDIA_EXPEDIENTE_UNIVERSITA", "etapa_ordinal_num",
    "vinculacion_previa",
]

FEATURES_MASTER = [
    "NU_NOTA_MEDIA_1_BACH__PC", "PAID_PERCENT",
    "CH_MATRICULA_SUJETA_BECA", "CU_IMPORTE_TOTAL",
    "NU_PREFERENCIA", "PL_SITUACION_SOCIO_ECONOMICA",
    "tiempo_etapa_dias", "tiempo_entre_etapas_dias",
    "NU_MEDIA_EXPEDIENTE_UNIVERSITA", "vinculacion_previa",
    "etapa_ordinal_num",
]


def make_df_features(features: list, n: int = 5) -> pd.DataFrame:
    """
    DataFrame sintético con valores realistas para las features dadas.
    Todos los valores son numéricos válidos — sin NaN.
    """
    data = {}
    for col in features:
        if "NOTA" in col or "ADMISION" in col or "EXPEDIENTE" in col:
            data[col] = np.round(np.random.uniform(5.0, 10.0, n), 2)
        elif "PAID" in col or "PORCENTAJE" in col:
            data[col] = np.round(np.random.uniform(0.0, 1.0, n), 2)
        elif "IMPORTE" in col or "precio" in col.lower():
            data[col] = np.round(np.random.uniform(3000, 8000, n), 2)
        elif "PREFERENCIA" in col:
            data[col] = np.random.randint(1, 4, n).astype(float)
        elif "SITUACION" in col:
            data[col] = np.random.choice([1, 2, 3, 4, 5], n).astype(float)
        elif "dias" in col or "tiempo" in col:
            data[col] = np.random.randint(1, 180, n).astype(float)
        elif "acum" in col or "asistencias" in col or "solicitudes" in col:
            data[col] = np.random.randint(0, 5, n).astype(float)
        elif "ordinal" in col or "etapa" in col:
            data[col] = np.random.randint(0, 10, n).astype(float)
        elif "vinculacion" in col or "BECA" in col:
            data[col] = np.random.randint(0, 2, n).astype(float)
        else:
            data[col] = np.round(np.random.uniform(0.0, 1.0, n), 2)
    return pd.DataFrame(data)


def make_df_ids(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "ID": [f"OPP{i:03d}" for i in range(n)],
    })


# ─── Tests de predecir() ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_grado():
    return load_model(str(MODELS_DIR / "modelo_final_grado"), verbose=False)


@pytest.fixture(scope="module")
def model_master():
    return load_model(str(MODELS_DIR / "modelo_final_master"), verbose=False)


def test_predecir_grado_columnas(model_grado):
    """predecir() con modelo grado genera prob_matricula_real y confianza_modelo."""
    df = make_df_features(FEATURES_GRADO, n=5)
    result = predecir(model_grado, df)
    assert "prob_matricula_real" in result.columns
    assert "confianza_modelo" in result.columns
    assert "prediction_label" in result.columns
    assert "prediction_score" in result.columns


def test_predecir_master_columnas(model_master):
    """predecir() con modelo master genera prob_matricula_real y confianza_modelo."""
    df = make_df_features(FEATURES_MASTER, n=5)
    result = predecir(model_master, df)
    assert "prob_matricula_real" in result.columns
    assert "confianza_modelo" in result.columns


def test_predecir_grado_probabilidad_en_rango(model_grado):
    """prob_matricula_real del modelo grado debe estar en [0, 1]."""
    df = make_df_features(FEATURES_GRADO, n=10)
    result = predecir(model_grado, df)
    assert (result["prob_matricula_real"] >= 0).all()
    assert (result["prob_matricula_real"] <= 1).all()


def test_predecir_master_probabilidad_en_rango(model_master):
    """prob_matricula_real del modelo master debe estar en [0, 1]."""
    df = make_df_features(FEATURES_MASTER, n=10)
    result = predecir(model_master, df)
    assert (result["prob_matricula_real"] >= 0).all()
    assert (result["prob_matricula_real"] <= 1).all()


def test_predecir_grado_confianza_en_rango(model_grado):
    """confianza_modelo del modelo grado debe estar en [0, 1]."""
    df = make_df_features(FEATURES_GRADO, n=10)
    result = predecir(model_grado, df)
    assert (result["confianza_modelo"] >= 0).all()
    assert (result["confianza_modelo"] <= 1).all()


def test_predecir_master_confianza_en_rango(model_master):
    """confianza_modelo del modelo master debe estar en [0, 1]."""
    df = make_df_features(FEATURES_MASTER, n=10)
    result = predecir(model_master, df)
    assert (result["confianza_modelo"] >= 0).all()
    assert (result["confianza_modelo"] <= 1).all()


def test_predecir_grado_target_pred_binario(model_grado):
    """prediction_label del modelo grado debe ser 0 o 1."""
    df = make_df_features(FEATURES_GRADO, n=10)
    result = predecir(model_grado, df)
    assert set(result["prediction_label"].unique()).issubset({0, 1})


def test_predecir_master_target_pred_binario(model_master):
    """prediction_label del modelo master debe ser 0 o 1."""
    df = make_df_features(FEATURES_MASTER, n=10)
    result = predecir(model_master, df)
    assert set(result["prediction_label"].unique()).issubset({0, 1})


def test_predecir_prob_real_coherente_con_label_grado(model_grado):
    """
    Si prediction_label==1, prob_matricula_real == prediction_score.
    Si prediction_label==0, prob_matricula_real == 1 - prediction_score.
    """
    df = make_df_features(FEATURES_GRADO, n=20)
    result = predecir(model_grado, df)

    for _, row in result.iterrows():
        if row["prediction_label"] == 1:
            assert abs(row["prob_matricula_real"] - row["prediction_score"]) < 1e-6
        else:
            assert abs(row["prob_matricula_real"] - (1 - row["prediction_score"])) < 1e-6


def test_predecir_confianza_formula_grado(model_grado):
    """confianza_modelo = abs(prob_matricula_real - 0.5) * 2."""
    df = make_df_features(FEATURES_GRADO, n=20)
    result = predecir(model_grado, df)
    expected_confianza = (result["prob_matricula_real"] - 0.5).abs() * 2
    assert (abs(result["confianza_modelo"] - expected_confianza) < 1e-6).all()


# ─── Tests de construir_resultado() ───────────────────────────────────────────

def test_construir_resultado_columnas(model_grado):
    """construir_resultado() genera el DataFrame con columnas correctas."""
    df_feat = make_df_features(FEATURES_GRADO, n=5)
    df_ids = make_df_ids(5)
    preds = predecir(model_grado, df_feat)
    resultado = construir_resultado(df_ids, preds, "grado", datetime.now())

    expected = {"OPP_ID", "PROBABILIDAD", "TARGET_PRED", "CONFIANZA", "MODELO", "FECHA_PRED"}
    assert expected.issubset(set(resultado.columns))


def test_construir_resultado_probabilidad_en_rango(model_grado):
    """PROBABILIDAD en [0, 1]."""
    df_feat = make_df_features(FEATURES_GRADO, n=10)
    df_ids = make_df_ids(10)
    preds = predecir(model_grado, df_feat)
    resultado = construir_resultado(df_ids, preds, "grado", datetime.now())
    assert (resultado["PROBABILIDAD"] >= 0).all()
    assert (resultado["PROBABILIDAD"] <= 1).all()


def test_construir_resultado_target_pred_binario(model_master):
    """TARGET_PRED debe ser 0 o 1."""
    df_feat = make_df_features(FEATURES_MASTER, n=10)
    df_ids = make_df_ids(10)
    preds = predecir(model_master, df_feat)
    resultado = construir_resultado(df_ids, preds, "master", datetime.now())
    assert set(resultado["TARGET_PRED"].unique()).issubset({0, 1})


def test_construir_resultado_campo_modelo_grado(model_grado):
    """El campo MODELO debe ser 'grado'."""
    df_feat = make_df_features(FEATURES_GRADO, n=3)
    df_ids = make_df_ids(3)
    preds = predecir(model_grado, df_feat)
    resultado = construir_resultado(df_ids, preds, "grado", datetime.now())
    assert (resultado["MODELO"] == "grado").all()


def test_construir_resultado_campo_modelo_master(model_master):
    """El campo MODELO debe ser 'master'."""
    df_feat = make_df_features(FEATURES_MASTER, n=3)
    df_ids = make_df_ids(3)
    preds = predecir(model_master, df_feat)
    resultado = construir_resultado(df_ids, preds, "master", datetime.now())
    assert (resultado["MODELO"] == "master").all()


def test_construir_resultado_opp_id_correcto(model_grado):
    """OPP_ID debe coincidir con los IDs del df_ids."""
    df_feat = make_df_features(FEATURES_GRADO, n=4)
    df_ids = make_df_ids(4)
    preds = predecir(model_grado, df_feat)
    resultado = construir_resultado(df_ids, preds, "grado", datetime.now())
    assert list(resultado["OPP_ID"]) == list(df_ids["ID"])


def test_construir_resultado_confianza_en_rango(model_master):
    """CONFIANZA debe estar en [0, 1]."""
    df_feat = make_df_features(FEATURES_MASTER, n=10)
    df_ids = make_df_ids(10)
    preds = predecir(model_master, df_feat)
    resultado = construir_resultado(df_ids, preds, "master", datetime.now())
    assert (resultado["CONFIANZA"] >= 0).all()
    assert (resultado["CONFIANZA"] <= 1).all()


# ─── Test de cargar_modelo() ──────────────────────────────────────────────────

def test_cargar_modelo_grado():
    """cargar_modelo('grado') devuelve el pipeline de PyCaret."""
    model = cargar_modelo("grado")
    assert model is not None
    assert hasattr(model, "predict")


def test_cargar_modelo_master():
    """cargar_modelo('master') devuelve el pipeline de PyCaret."""
    model = cargar_modelo("master")
    assert model is not None
    assert hasattr(model, "predict")
