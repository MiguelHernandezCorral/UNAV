"""
Fase A - Tests de PREDICCIONES_V2 + explicabilidad SHAP.

Grupos:
    1. explainer.py — sin PyCaret real (usa sklearn Pipeline mínimo)
    2. predictor.py::construir_resultado_v2() — sin Oracle
    3. predictor.py::guardar_en_oracle_v2() — mock OracleConnector
    4. Integration — modelo PKL real (marcado @pytest.mark.integration)

Sin conexión Oracle. Tests reales para lógica pura; mocks solo para Oracle.
"""
import json
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR      = Path(__file__).parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from explainer import (
    generar_explicacion,
    _detectar_tipo_clasificador,
    _normalizar_shap_clase1,
    _formatear_fila,
)
from predictor import (
    construir_resultado_v2,
    guardar_en_oracle_v2,
    _version_modelo,
)

# ─── Fixtures de datos sintéticos ─────────────────────────────────────────────

FEATURES = ["etapa_ordinal_num", "PAID_PERCENT", "vinculacion_previa",
            "tiempo_etapa_dias", "num_asistencias_acum"]

N = 5


def make_df_model(n: int = N) -> pd.DataFrame:
    return pd.DataFrame({
        "etapa_ordinal_num":   np.random.randint(0, 10, n).astype(float),
        "PAID_PERCENT":        np.random.uniform(0, 1, n),
        "vinculacion_previa":  np.random.randint(0, 2, n).astype(float),
        "tiempo_etapa_dias":   np.random.randint(1, 180, n).astype(float),
        "num_asistencias_acum": np.random.randint(0, 5, n).astype(float),
    })


def make_df_ids(n: int = N, include_target: bool = True) -> pd.DataFrame:
    df = pd.DataFrame({
        "ID":            [f"OPP{i:03d}" for i in range(n)],
        "PL_Etapa__c":   ["Validación"] * n,
        "PL_Subetapa__c":["Completa"] * n,
    })
    if include_target:
        df["target"] = np.random.randint(0, 2, n)
    return df


def make_preds(n: int = N) -> pd.DataFrame:
    labels = np.random.randint(0, 2, n)
    scores = np.random.uniform(0.5, 1.0, n)
    prob_real = np.where(labels == 1, scores, 1 - scores)
    confianza = np.abs(prob_real - 0.5) * 2
    return pd.DataFrame({
        "prediction_label": labels,
        "prediction_score": scores,
        "prob_matricula_real": prob_real,
        "confianza_modelo": confianza,
    })


def make_sklearn_pipeline(n_features: int = 5):
    """Pipeline sklearn mínimo: StandardScaler + LogisticRegression."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    X = np.random.rand(50, n_features)
    y = np.random.randint(0, 2, 50)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(random_state=42, max_iter=200)),
    ])
    pipe.fit(X, y)
    return pipe


# ─── Tests: explainer._detectar_tipo_clasificador ─────────────────────────────

def test_detectar_tipo_tree_lightgbm():
    try:
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier()
        assert _detectar_tipo_clasificador(clf) == "tree"
    except ImportError:
        pytest.skip("LightGBM no disponible")


def test_detectar_tipo_linear():
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    assert _detectar_tipo_clasificador(clf) == "linear"


def test_detectar_tipo_generic():
    class MiModelo:
        pass
    assert _detectar_tipo_clasificador(MiModelo()) == "generic"


# ─── Tests: explainer._normalizar_shap_clase1 ─────────────────────────────────

def test_normalizar_shap_clase1_lista():
    sv0 = np.zeros((5, 3))
    sv1 = np.ones((5, 3)) * 0.5
    result = _normalizar_shap_clase1([sv0, sv1])
    assert result.shape == (5, 3)
    assert np.allclose(result, 0.5)


def test_normalizar_shap_clase1_3d():
    sv = np.random.rand(5, 3, 2)
    result = _normalizar_shap_clase1(sv)
    assert result.shape == (5, 3)
    assert np.allclose(result, sv[:, :, 1])


def test_normalizar_shap_clase1_2d():
    sv = np.random.rand(5, 3)
    result = _normalizar_shap_clase1(sv)
    assert result.shape == (5, 3)


# ─── Tests: explainer._formatear_fila ─────────────────────────────────────────

def test_formatear_fila_estructura():
    shap_row = np.array([0.3, -0.1, 0.5, 0.0, -0.2])
    names    = ["a", "b", "c", "d", "e"]
    vals     = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    resultado = _formatear_fila(shap_row, names, vals, top_n=3)
    assert "top_features" in resultado
    assert len(resultado["top_features"]) == 3
    for item in resultado["top_features"]:
        assert {"feature", "valor", "impacto", "efecto"} <= item.keys()


def test_formatear_fila_efecto_coherente():
    shap_row = np.array([0.3, -0.1, 0.5, 0.0, -0.2])
    names    = ["a", "b", "c", "d", "e"]
    vals     = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    resultado = _formatear_fila(shap_row, names, vals, top_n=5)
    for item in resultado["top_features"]:
        if item["impacto"] >= 0:
            assert item["efecto"] == "positivo"
        else:
            assert item["efecto"] == "negativo"


def test_formatear_fila_top_3_son_mayores():
    shap_row = np.array([0.1, 0.5, -0.3, 0.8, -0.05])
    names    = ["a", "b", "c", "d", "e"]
    vals     = pd.Series([0.0] * 5)
    resultado = _formatear_fila(shap_row, names, vals, top_n=3)
    features_sel = {item["feature"] for item in resultado["top_features"]}
    # Los 3 mayores en |shap| son d(0.8), b(0.5), c(0.3)
    assert features_sel == {"d", "b", "c"}


# ─── Tests: generar_explicacion ────────────────────────────────────────────────

def test_generar_explicacion_returns_series():
    pipe = make_sklearn_pipeline(n_features=5)
    df   = make_df_model(n=5)
    result = generar_explicacion(pipe, df)
    assert isinstance(result, pd.Series)
    assert len(result) == 5


def test_generar_explicacion_json_parseable():
    pipe = make_sklearn_pipeline(n_features=5)
    df   = make_df_model(n=5)
    result = generar_explicacion(pipe, df)
    for val in result:
        parsed = json.loads(val)
        assert isinstance(parsed, dict)


def test_generar_explicacion_top3_features():
    pipe = make_sklearn_pipeline(n_features=5)
    df   = make_df_model(n=5)
    result = generar_explicacion(pipe, df)
    for val in result:
        parsed = json.loads(val)
        if "top_features" in parsed:
            assert len(parsed["top_features"]) == 3


def test_generar_explicacion_efecto_coherente():
    pipe = make_sklearn_pipeline(n_features=5)
    df   = make_df_model(n=5)
    result = generar_explicacion(pipe, df)
    for val in result:
        parsed = json.loads(val)
        for feat in parsed.get("top_features", []):
            if feat["impacto"] >= 0:
                assert feat["efecto"] == "positivo"
            else:
                assert feat["efecto"] == "negativo"


def test_generar_explicacion_sin_shap_no_crash():
    """Si SHAP no está disponible el fallback es '{}' sin excepción."""
    pipe = make_sklearn_pipeline(n_features=5)
    df   = make_df_model(n=5)

    with patch("explainer.SHAP_AVAILABLE", False):
        result = generar_explicacion(pipe, df)

    assert isinstance(result, pd.Series)
    assert len(result) == 5
    for val in result:
        assert val == "{}"


def test_generar_explicacion_df_vacio():
    """DataFrame vacío devuelve Serie vacía."""
    pipe = make_sklearn_pipeline(n_features=5)
    df   = pd.DataFrame(columns=FEATURES)
    result = generar_explicacion(pipe, df)
    assert isinstance(result, pd.Series)
    assert len(result) == 0


@pytest.mark.integration
def test_generar_explicacion_real_grado():
    """Integración: modelo PKL real de grado + datos sintéticos."""
    try:
        from pycaret.classification import load_model
    except ImportError:
        pytest.skip("PyCaret no disponible")

    model_path = PROJECT_ROOT / "models" / "modelo_final_grado"
    if not model_path.with_suffix(".pkl").exists():
        pytest.skip("modelo_final_grado.pkl no encontrado")

    model = load_model(str(model_path), verbose=False)
    features = [f for f in model.feature_names_in_ if f != "target"]
    df = pd.DataFrame({f: np.random.uniform(0, 1, 5) for f in features})

    result = generar_explicacion(model, df)
    assert isinstance(result, pd.Series)
    assert len(result) == 5
    for val in result:
        parsed = json.loads(val)
        # Puede ser {} si SHAP falla, pero no debe lanzar excepción
        assert isinstance(parsed, dict)


# ─── Tests: construir_resultado_v2 ────────────────────────────────────────────

def test_construir_resultado_v2_columnas():
    df_ids = make_df_ids(N)
    preds  = make_preds(N)
    result = construir_resultado_v2(df_ids, preds, "grado", datetime.now())
    expected = {
        "OPP_ID_ETAPA_COMP", "OPP_ID", "ETAPA", "SUBETAPA",
        "TARGET_REAL", "TARGET_PRED", "PROBABILIDAD", "CONFIANZA",
        "MODELO", "EXPLICACION", "FECHA_PRED", "FECHA_ACTUALIZACION",
    }
    assert expected.issubset(set(result.columns))


def test_construir_resultado_v2_pk_format():
    df_ids = make_df_ids(N)
    preds  = make_preds(N)
    result = construir_resultado_v2(df_ids, preds, "grado", datetime.now())
    for i, row in result.iterrows():
        expected_pk = f"{row['OPP_ID']}__{row['ETAPA']}__{row['SUBETAPA']}"
        assert row["OPP_ID_ETAPA_COMP"] == expected_pk


def test_construir_resultado_v2_target_real_presente():
    df_ids = make_df_ids(N, include_target=True)
    preds  = make_preds(N)
    result = construir_resultado_v2(df_ids, preds, "grado", datetime.now())
    assert result["TARGET_REAL"].notna().all()


def test_construir_resultado_v2_target_real_none_si_ausente():
    df_ids = make_df_ids(N, include_target=False)
    preds  = make_preds(N)
    result = construir_resultado_v2(df_ids, preds, "grado", datetime.now())
    assert result["TARGET_REAL"].isna().all()


def test_construir_resultado_v2_modelo_versionado():
    df_ids = make_df_ids(N)
    preds  = make_preds(N)
    result = construir_resultado_v2(df_ids, preds, "grado", datetime.now())
    # MODELO debe ser 'grado_v1' si solo existe modelo_final_grado.pkl
    assert result["MODELO"].str.startswith("grado").all()
    assert result["MODELO"].str.match(r"grado_v\d+").all()


def test_construir_resultado_v2_explicacion_none_fallback():
    """Si explicaciones=None → EXPLICACION debe ser '{}'."""
    df_ids = make_df_ids(N)
    preds  = make_preds(N)
    result = construir_resultado_v2(df_ids, preds, "grado", datetime.now(), explicaciones=None)
    assert (result["EXPLICACION"] == "{}").all()


def test_construir_resultado_v2_con_explicaciones():
    df_ids = make_df_ids(N)
    preds  = make_preds(N)
    expl   = pd.Series(['{"top_features": []}'] * N)
    result = construir_resultado_v2(df_ids, preds, "grado", datetime.now(), explicaciones=expl)
    assert (result["EXPLICACION"] == '{"top_features": []}').all()


def test_construir_resultado_v2_probabilidad_rango():
    df_ids = make_df_ids(N)
    preds  = make_preds(N)
    result = construir_resultado_v2(df_ids, preds, "grado", datetime.now())
    assert (result["PROBABILIDAD"] >= 0).all()
    assert (result["PROBABILIDAD"] <= 1).all()


def test_construir_resultado_v2_target_pred_binario():
    df_ids = make_df_ids(N)
    preds  = make_preds(N)
    result = construir_resultado_v2(df_ids, preds, "master", datetime.now())
    assert set(result["TARGET_PRED"].unique()).issubset({0, 1})


# ─── Tests: guardar_en_oracle_v2 (mock Oracle) ────────────────────────────────

def test_guardar_v2_usa_upsert():
    """guardar_en_oracle_v2 debe llamar a upsert_records con pk=OPP_ID_ETAPA_COMP."""
    df_ids = make_df_ids(N)
    preds  = make_preds(N)
    df_res = construir_resultado_v2(df_ids, preds, "grado", datetime.now())

    mock_conn = MagicMock()
    # OracleConnector se importa dentro de guardar_en_oracle_v2 desde oracle_connector
    with patch("oracle_connector.OracleConnector", return_value=mock_conn):
        guardar_en_oracle_v2(df_res, save_hist=False)

    mock_conn.upsert_records.assert_called_once()
    call_args = mock_conn.upsert_records.call_args
    # Puede llamarse como upsert_records(records, table, pk_col=...) o posicional
    pk_usado = call_args[1].get("pk_col") or (call_args[0][2] if len(call_args[0]) >= 3 else None)
    assert pk_usado == "OPP_ID_ETAPA_COMP"


def test_guardar_v2_save_hist_llama_insert():
    """Con save_hist=True también debe llamar a insert_records."""
    df_ids = make_df_ids(N)
    preds  = make_preds(N)
    df_res = construir_resultado_v2(df_ids, preds, "grado", datetime.now())

    mock_conn = MagicMock()
    with patch("oracle_connector.OracleConnector", return_value=mock_conn):
        guardar_en_oracle_v2(df_res, save_hist=True)

    mock_conn.upsert_records.assert_called_once()
    mock_conn.insert_records.assert_called_once()


def test_guardar_v2_sin_hist_no_llama_insert():
    """Con save_hist=False NO debe llamar a insert_records."""
    df_ids = make_df_ids(N)
    preds  = make_preds(N)
    df_res = construir_resultado_v2(df_ids, preds, "grado", datetime.now())

    mock_conn = MagicMock()
    with patch("oracle_connector.OracleConnector", return_value=mock_conn):
        guardar_en_oracle_v2(df_res, save_hist=False)

    mock_conn.insert_records.assert_not_called()


# ─── Tests: _version_modelo ───────────────────────────────────────────────────

def test_version_modelo_grado_v1():
    """Si existe modelo_final_grado.pkl → 'grado_v1'."""
    modelo_path = PROJECT_ROOT / "models" / "modelo_final_grado.pkl"
    if not modelo_path.exists():
        pytest.skip("modelo_final_grado.pkl no encontrado")
    version = _version_modelo("grado")
    assert version == "grado_v1"


def test_version_modelo_master_v1():
    modelo_path = PROJECT_ROOT / "models" / "modelo_final_master.pkl"
    if not modelo_path.exists():
        pytest.skip("modelo_final_master.pkl no encontrado")
    version = _version_modelo("master")
    assert version == "master_v1"
