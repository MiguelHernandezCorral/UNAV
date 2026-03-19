"""
src/explainer.py
================
Fase A · Explicabilidad SHAP por fila para los modelos PyCaret.

Genera para cada predicción un JSON con las 3 variables que más influyeron
en la probabilidad de matrícula, indicando su valor real y su impacto
(positivo = aumenta la probabilidad, negativo = la reduce).

Uso:
    from explainer import generar_explicacion
    explicaciones = generar_explicacion(model, df_model)  # pd.Series de JSON strings

JSON de salida por fila:
    {
      "top_features": [
        {"feature": "etapa_ordinal_num",  "valor": 8,   "impacto":  0.23, "efecto": "positivo"},
        {"feature": "PAID_PERCENT",        "valor": 0.0, "impacto": -0.18, "efecto": "negativo"},
        {"feature": "vinculacion_previa",  "valor": 1,   "impacto":  0.09, "efecto": "positivo"}
      ]
    }

Si SHAP no está disponible o falla → devuelve Serie de strings vacíos "{}".
Nunca lanza excepción que bloquee el pipeline de predicciones.
"""

import json
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ─── Importación opcional de SHAP ─────────────────────────────────────────────

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None  # type: ignore[assignment]
    SHAP_AVAILABLE = False
    logger.warning("SHAP no está instalado. Las explicaciones estarán vacías.")


# ─── Funciones internas ────────────────────────────────────────────────────────

def _detectar_tipo_clasificador(estimator) -> str:
    """
    Devuelve el tipo de clasificador para elegir el Explainer correcto.

    Returns:
        'tree'    → LightGBM, XGBoost, CatBoost, RandomForest, GBM, ExtraTrees
        'linear'  → LogisticRegression, Ridge, LinearSVC
        'generic' → cualquier otro (usa shap.Explainer, más lento)
    """
    module = type(estimator).__module__ or ""
    name   = type(estimator).__name__   or ""

    tree_keywords = ("lightgbm", "xgboost", "catboost", "forest", "tree",
                     "gradient", "extra", "hist")
    linear_keywords = ("logistic", "ridge", "linear", "sgd", "lasso")

    combined = (module + name).lower()
    if any(kw in combined for kw in tree_keywords):
        return "tree"
    if any(kw in combined for kw in linear_keywords):
        return "linear"
    return "generic"


def _normalizar_shap_clase1(shap_values) -> np.ndarray:
    """
    Normaliza la salida de shap_values a un array 2D (n_filas, n_features)
    para la clase 1 (matrícula = True).

    PyCaret / LightGBM puede devolver:
    - list de dos arrays [clase_0, clase_1]
    - array 3D (n, features, 2)
    - array 2D directo (binario)
    """
    if isinstance(shap_values, list):
        # [clase_0_array, clase_1_array]
        return np.array(shap_values[1])

    sv = np.array(shap_values)
    if sv.ndim == 3:
        # (n, features, 2) → tomar clase 1
        return sv[:, :, 1]

    # 2D → ya es la salida directa (clase 1 en clasificación binaria)
    return sv


def _formatear_fila(
    shap_row: np.ndarray,
    feature_names: list[str],
    feature_values: pd.Series,
    top_n: int = 3,
) -> dict:
    """
    Construye el dict con las top_n features más influyentes para una fila.

    Args:
        shap_row:       Array 1D de SHAP values para esta fila.
        feature_names:  Nombres de las features (orden = orden del array).
        feature_values: Serie con los valores originales (df_model.iloc[i]).
        top_n:          Número de features a incluir (default: 3).

    Returns:
        dict con clave 'top_features'.
    """
    # Índices ordenados por |impacto| descendente
    top_idx = np.argsort(np.abs(shap_row))[::-1][:top_n]

    features = []
    for idx in top_idx:
        impacto = float(round(shap_row[idx], 4))
        valor_raw = feature_values.iloc[idx]

        # Convertir a tipo serializable
        if isinstance(valor_raw, (np.integer,)):
            valor = int(valor_raw)
        elif isinstance(valor_raw, (np.floating, float)):
            valor = float(round(float(valor_raw), 4))
        elif isinstance(valor_raw, bool):
            valor = int(valor_raw)
        else:
            valor = valor_raw

        features.append({
            "feature": feature_names[idx],
            "valor":   valor,
            "impacto": impacto,
            "efecto":  "positivo" if impacto >= 0 else "negativo",
        })

    return {"top_features": features}


# ─── Función pública ───────────────────────────────────────────────────────────

def generar_explicacion(model, df_model: pd.DataFrame) -> pd.Series:
    """
    Calcula SHAP values y devuelve una Serie de JSON strings (uno por fila).

    Usa TreeExplainer para modelos basados en árboles (LightGBM, XGBoost…),
    LinearExplainer para modelos lineales y shap.Explainer genérico para el resto.

    Args:
        model:    Pipeline PyCaret cargado (sklearn Pipeline completo).
        df_model: DataFrame preprocesado, el mismo que se pasa a predict_model().
                  Debe tener exactamente las columnas de model.feature_names_in_.

    Returns:
        pd.Series de strings JSON, indexada igual que df_model.
        Si SHAP no está disponible o falla → Serie de strings vacíos "{}".
    """
    n = len(df_model)
    fallback = pd.Series(['{}'] * n, index=df_model.index)

    if not SHAP_AVAILABLE:
        logger.warning("SHAP no disponible — devolviendo explicaciones vacías.")
        return fallback

    if n == 0:
        return fallback

    try:
        # 1. Extraer estimador final del pipeline PyCaret
        final_estimator = model.steps[-1][1]

        # 2. Transformar datos a través del pipeline (sin el estimador final)
        #    model[:-1] crea un sub-pipeline con todos los pasos excepto el último
        try:
            X_transformed = model[:-1].transform(df_model)
        except Exception as e_transform:
            logger.warning(
                "No se pudo transformar df_model con model[:-1]: %s — usando df_model.values",
                e_transform,
            )
            X_transformed = df_model.values

        # 3. Nombres de features (recuperar desde df_model)
        feature_names = df_model.columns.tolist()

        # Validar alineación de dimensiones
        if X_transformed.shape[1] != len(feature_names):
            logger.warning(
                "Dimensión transformada (%d) ≠ features (%d) — "
                "las explicaciones pueden ser inexactas.",
                X_transformed.shape[1], len(feature_names),
            )
            # Ajustar nombres si es posible
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        # 4. Crear Explainer según tipo de clasificador
        tipo = _detectar_tipo_clasificador(final_estimator)
        logger.info("SHAP: tipo de clasificador detectado = '%s'", tipo)

        if tipo == "tree":
            explainer = shap.TreeExplainer(final_estimator)
        elif tipo == "linear":
            explainer = shap.LinearExplainer(final_estimator, X_transformed)
        else:
            logger.info("SHAP: usando Explainer genérico (puede ser lento).")
            explainer = shap.Explainer(final_estimator, X_transformed)

        # 5. Calcular SHAP values
        raw = explainer.shap_values(X_transformed)
        sv = _normalizar_shap_clase1(raw)

        if sv.shape[0] != n:
            raise ValueError(
                f"SHAP devolvió {sv.shape[0]} filas pero df_model tiene {n}"
            )

        # 6. Construir JSON por fila
        resultados = []
        for i in range(n):
            fila_dict = _formatear_fila(
                shap_row=sv[i],
                feature_names=feature_names,
                feature_values=df_model.iloc[i],
            )
            resultados.append(json.dumps(fila_dict, ensure_ascii=False))

        logger.info("Explicaciones SHAP generadas: %d filas.", n)
        return pd.Series(resultados, index=df_model.index)

    except Exception as exc:
        logger.warning(
            "Error al generar explicaciones SHAP: %s — devolviendo vacías.", exc,
            exc_info=True,
        )
        return fallback
