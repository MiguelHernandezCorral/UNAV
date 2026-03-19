"""
src/predictor.py
================
Fase 4 · Predicciones de matrícula con modelos PyCaret preentrenados.

Flujo:
    1. Carga DATASET_LIMPIO desde Oracle (via preprocessor.load_dataset_limpio)
    2. Para cada segmento (grado, master):
       a. Preprocesa con preprocessor.preprocess()
       b. Carga modelo PyCaret desde models/
       c. Predice con predict_model()
       d. Calcula prob_matricula_real y confianza_modelo
       e. Construye tabla de resultados
    3. Une los resultados de ambos segmentos
    4. Inserta en Oracle tabla PREDICCIONES (historial completo, sin upsert)

Uso directo:
    python src/predictor.py

Resultado en Oracle:
    Tabla PREDICCIONES con columnas:
        OPP_ID, PROBABILIDAD, TARGET_PRED, CONFIANZA, MODELO, FECHA_PRED
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Asegurar que src/ está en el path
SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logger = logging.getLogger(__name__)

TIPOS = ["grado", "master"]
PREDICCIONES_TABLE = "PREDICCIONES"

# ─── Imports opcionales (se mockean en tests) ─────────────────────────────────

try:
    from pycaret.classification import load_model, predict_model
    PYCARET_AVAILABLE = True
except ImportError:
    load_model = None       # type: ignore[assignment]
    predict_model = None    # type: ignore[assignment]
    PYCARET_AVAILABLE = False

try:
    from preprocessor import load_dataset_limpio, preprocess
except ImportError:
    load_dataset_limpio = None  # type: ignore[assignment]
    preprocess = None           # type: ignore[assignment]


# ─── Funciones principales ────────────────────────────────────────────────────

def cargar_modelo(tipo: str):
    """
    Carga el modelo PyCaret para el segmento indicado.

    Args:
        tipo: 'grado' o 'master'

    Returns:
        Pipeline PyCaret cargado.
    """
    ruta = str(MODELS_DIR / f"modelo_final_{tipo}")
    logger.info(f"Cargando modelo: {ruta}.pkl")
    return load_model(ruta, verbose=False)


def predecir(model, df_model: pd.DataFrame) -> pd.DataFrame:
    """
    Genera predicciones y calcula probabilidades y confianza.

    Args:
        model:    Pipeline PyCaret cargado.
        df_model: DataFrame preprocesado (safe_cols).

    Returns:
        DataFrame con columnas:
            prediction_label, prediction_score,
            prob_matricula_real, confianza_modelo
    """
    preds = predict_model(model, data=df_model, verbose=False)

    # Probabilidad real de matrícula (P(y=1) siempre en [0, 1])
    preds["prob_matricula_real"] = preds.apply(
        lambda x: x["prediction_score"]
        if x["prediction_label"] == 1
        else (1 - x["prediction_score"]),
        axis=1,
    )

    # Confianza: distancia normalizada al punto de indecisión (0.5)
    # 0 = modelo indeciso, 1 = certeza máxima
    preds["confianza_modelo"] = (preds["prob_matricula_real"] - 0.5).abs() * 2

    return preds


def construir_resultado(
    df_ids: pd.DataFrame,
    preds: pd.DataFrame,
    tipo: str,
    fecha: datetime,
) -> pd.DataFrame:
    """
    Construye la tabla de resultados lista para insertar en Oracle.

    Args:
        df_ids: DataFrame con columna ID (y opcionalmente target) de auditoría.
        preds:  DataFrame de predicciones de PyCaret.
        tipo:   'grado' o 'master'.
        fecha:  Timestamp de la ejecución.

    Returns:
        DataFrame con columnas: OPP_ID, PROBABILIDAD, TARGET_PRED, CONFIANZA, MODELO, FECHA_PRED
    """
    resultado = pd.DataFrame({
        "OPP_ID": df_ids["ID"].values,
        "PROBABILIDAD": preds["prob_matricula_real"].values,
        "TARGET_PRED": preds["prediction_label"].astype(int).values,
        "CONFIANZA": preds["confianza_modelo"].values,
        "MODELO": tipo,
        "FECHA_PRED": fecha,
    })
    return resultado


def guardar_en_oracle(df_resultado: pd.DataFrame) -> None:
    """
    Inserta el DataFrame de resultados en la tabla PREDICCIONES de Oracle.

    Usa INSERT (no UPSERT) para mantener historial completo por ejecución.
    La tabla se crea automáticamente si no existe.

    Args:
        df_resultado: DataFrame con columnas de PREDICCIONES.
    """
    from oracle_connector import OracleConnector  # noqa: PLC0415

    conn = OracleConnector()
    records = df_resultado.to_dict("records")

    # Convertir FECHA_PRED a string ISO para compatibilidad con _infer_ora_type
    for rec in records:
        if isinstance(rec.get("FECHA_PRED"), datetime):
            rec["FECHA_PRED"] = rec["FECHA_PRED"].isoformat()

    logger.info(f"Insertando {len(records)} registros en {PREDICCIONES_TABLE}...")
    conn.insert_records(records, PREDICCIONES_TABLE)
    logger.info("Inserción completada.")


def run_predictions(
    save_to_oracle: bool = True,
    return_df: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Ejecuta el pipeline completo de predicciones para Grado y Máster.

    Args:
        save_to_oracle: Si True, inserta resultados en Oracle (default: True).
        return_df:      Si True, devuelve el DataFrame combinado de resultados.

    Returns:
        DataFrame combinado si return_df=True, None en caso contrario.
    """
    fecha_ejecucion = datetime.now()
    logger.info(f"Iniciando predicciones — {fecha_ejecucion.isoformat()}")

    # Cargar dataset una sola vez
    df_raw = load_dataset_limpio()

    resultados = []

    for tipo in TIPOS:
        logger.info(f"--- Procesando segmento: {tipo} ---")

        # Cargar modelo primero para obtener las features exactas que necesita
        model = cargar_modelo(tipo)
        # feature_names_in_ incluye 'target' del setup original — lo excluimos
        model_features = [f for f in model.feature_names_in_ if f != "target"]

        # Preprocesado usando las features exactas del modelo (modo inferencia)
        df_model, _, df_ids = preprocess(df_raw, tipo, model_features=model_features)

        if df_model.empty:
            logger.warning(f"No hay filas para el segmento '{tipo}' — se omite.")
            continue

        # Predicción
        preds = predecir(model, df_model)

        # Construcción tabla resultado
        df_resultado = construir_resultado(df_ids, preds, tipo, fecha_ejecucion)
        resultados.append(df_resultado)

        n_matricula = int((df_resultado["TARGET_PRED"] == 1).sum())
        logger.info(
            f"Segmento '{tipo}': {len(df_resultado)} predicciones | "
            f"Matrícula prevista: {n_matricula} ({n_matricula/len(df_resultado)*100:.1f}%)"
        )

    if not resultados:
        logger.error("No se generaron predicciones para ningún segmento.")
        return None

    df_final = pd.concat(resultados, ignore_index=True)
    logger.info(f"Total predicciones: {len(df_final)}")

    if save_to_oracle:
        guardar_en_oracle(df_final)

    if return_df:
        return df_final

    return None


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_predictions(save_to_oracle=True)
