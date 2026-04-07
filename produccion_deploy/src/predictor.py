"""
src/predictor.py
================
Fase 3 · Predicciones de matrícula con modelos PyCaret preentrenados.

Flujo original (run_predictions):
    1. Carga DATASET_LIMPIO desde Oracle
    2. Para cada segmento (grado, master): preprocesa, predice, construye resultado
    3. Inserta en Oracle tabla PREDICCIONES (INSERT histórico)

Flujo enriquecido (run_predictions_v2):
    1. Carga DATASET_LIMPIO desde Oracle
    2. Para cada segmento:
       a. Preprocesa con preprocessor.preprocess()
       b. Carga modelo PyCaret desde models/
       c. Predice con predict_model()
       d. Calcula prob_matricula_real y confianza_modelo
       e. Genera explicaciones SHAP (top-3 features) via explainer.py
       f. Construye tabla PMAT_PREDICTION (con etapa, subetapa, target_real, explicacion)
    3. UPSERT en Oracle tabla PMAT_PREDICTION por clave (OPP_ID, ETAPA, SUBETAPA)
    4. Opcionalmente: INSERT en PMAT_PREDICTION_HIST para auditoría completa

Uso directo:
    python src/predictor.py             # usa run_predictions_v2 por defecto
    python src/predictor.py --legacy    # usa run_predictions (INSERT histórico)

Resultado en Oracle:
    PREDICCIONES         — historial completo (INSERT), esquema original
    PMAT_PREDICTION      — última predicción por (OPP_ID, ETAPA, SUBETAPA), con SHAP
    PMAT_PREDICTION_HIST — historial completo de PMAT_PREDICTION (opcional, --save-hist)
"""

import json
import logging
import re
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
PREDICCIONES_TABLE       = "PREDICCIONES"
PMAT_PREDICTION_TABLE    = "PMAT_PREDICTION"
PMAT_PREDICTION_HIST     = "PMAT_PREDICTION_HIST"

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


# ─── Funciones V2 ────────────────────────────────────────────────────────────

def _version_modelo(tipo: str) -> str:
    """
    Devuelve la versión del modelo a partir del nombre del PKL en MODELS_DIR.

    Ejemplos:
        modelo_final_grado.pkl    → 'grado_v1'
        modelo_final_grado_v2.pkl → 'grado_v2'
        modelo_final_master.pkl   → 'master_v1'
    """
    # Buscar todos los PKL del tipo (sin .bak)
    patron = re.compile(rf"modelo_final_{tipo}(?:_v(\d+))?\.pkl$")
    versiones = []
    for f in MODELS_DIR.glob(f"modelo_final_{tipo}*.pkl"):
        if ".bak" in f.name:
            continue
        m = patron.match(f.name)
        if m:
            v = int(m.group(1)) if m.group(1) else 1
            versiones.append(v)

    version = max(versiones) if versiones else 1
    return f"{tipo}_v{version}"


def construir_resultado_v2(
    df_ids: pd.DataFrame,
    preds: pd.DataFrame,
    tipo: str,
    fecha: datetime,
    explicaciones: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Construye la tabla PMAT_PREDICTION lista para upsert en Oracle.

    Args:
        df_ids:        DataFrame con columnas: ID, PL_Etapa__c, PL_Subetapa__c,
                       y opcionalmente 'target'.
        preds:         DataFrame de predicciones PyCaret (con prob_matricula_real,
                       confianza_modelo, prediction_label).
        tipo:          'grado' o 'master'.
        fecha:         Timestamp de la ejecución.
        explicaciones: pd.Series con JSON strings de SHAP (o None → '{}').

    Returns:
        DataFrame con columnas de PMAT_PREDICTION.
    """
    etapa   = df_ids.get("PL_Etapa__c",   pd.Series(["NA"] * len(df_ids))).fillna("NA")
    subetapa = df_ids.get("PL_Subetapa__c", pd.Series(["NA"] * len(df_ids))).fillna("NA")

    opp_id_etapa_comp = (
        df_ids["ID"].astype(str) + "__"
        + etapa.astype(str) + "__"
        + subetapa.astype(str)
    )

    expl = (
        explicaciones.values
        if explicaciones is not None
        else ['{}'] * len(df_ids)
    )

    target_real = (
        df_ids["target"].values
        if "target" in df_ids.columns
        else [None] * len(df_ids)
    )

    if "CreatedDate" in df_ids.columns:
        _ts = pd.to_datetime(df_ids["CreatedDate"], errors="coerce")
        fecha_inicio = [None if pd.isna(t) else t.to_pydatetime() for t in _ts]
    else:
        fecha_inicio = [None] * len(df_ids)

    resultado = pd.DataFrame({
        "OPP_ID_ETAPA_COMP":  opp_id_etapa_comp.values,
        "OPP_ID":             df_ids["ID"].values,
        "ETAPA":              etapa.values,
        "SUBETAPA":           subetapa.values,
        "FECHA_INICIO_ETAPA": fecha_inicio,
        "TARGET_REAL":        target_real,
        "TARGET_PRED":        preds["prediction_label"].astype(int).values,
        "PROBABILIDAD":       (preds["prob_matricula_real"] * 100).round(2).values,
        "CONFIANZA":          (preds["confianza_modelo"] * 100).round(2).values,
        "MODELO":             _version_modelo(tipo),
        "EXPLICACION":        expl,
        "FECHA_PRED":         fecha.isoformat(),
        "FECHA_ACTUALIZACION": fecha.isoformat(),
    })
    return resultado


def guardar_en_oracle_v2(df_resultado: pd.DataFrame, save_hist: bool = False) -> None:
    """
    Guarda PMAT_PREDICTION en Oracle usando UPSERT por OPP_ID_ETAPA_COMP.

    Estrategia:
        - UPSERT: 1 fila por (OPP_ID, ETAPA, SUBETAPA); se actualiza si cambia.
        - Si save_hist=True: INSERT también en PMAT_PREDICTION_HIST (auditoría completa).

    Args:
        df_resultado: DataFrame con columnas de PMAT_PREDICTION.
        save_hist:    Si True, también inserta en PMAT_PREDICTION_HIST.
    """
    from oracle_connector import OracleConnector  # noqa: PLC0415

    conn = OracleConnector()
    records = df_resultado.to_dict("records")

    # Migración automática: añadir FECHA_INICIO_ETAPA si la tabla ya existía sin ella
    if conn._table_exists("PMAT_PREDICTION"):
        conn.add_column_if_not_exists("PMAT_PREDICTION", "FECHA_INICIO_ETAPA", "TIMESTAMP")

    logger.info(
        f"Upsert {len(records)} registros en {PMAT_PREDICTION_TABLE} "
        f"(pk=OPP_ID_ETAPA_COMP)..."
    )
    conn.upsert_records(
        records,
        PMAT_PREDICTION_TABLE,
        pk_col="OPP_ID_ETAPA_COMP",
        compare_cols=["PROBABILIDAD"],
    )
    logger.info("Upsert completado.")

    # Actualizar vista con la última predicción por oportunidad
    _PMAT_PRED_ACTUAL_SQL = f"""
        SELECT p.OPP_ID, p.PROBABILIDAD, p.CONFIANZA,
               p.ETAPA, p.SUBETAPA, p.FECHA_INICIO_ETAPA, p.FECHA_ACTUALIZACION
        FROM "{conn.schema}"."PMAT_PREDICTION" p
        WHERE p.FECHA_INICIO_ETAPA = (
            SELECT MAX(p2.FECHA_INICIO_ETAPA)
            FROM "{conn.schema}"."PMAT_PREDICTION" p2
            WHERE p2.OPP_ID = p.OPP_ID
        )
    """
    conn.create_or_replace_view("PMAT_PRED_ACTUAL", _PMAT_PRED_ACTUAL_SQL)
    logger.info("Vista PMAT_PRED_ACTUAL actualizada.")

    if save_hist:
        logger.info(f"Insertando historial en {PMAT_PREDICTION_HIST}...")
        conn.insert_records(records, PMAT_PREDICTION_HIST)
        logger.info("Historial insertado.")


def run_predictions_v2(
    save_to_oracle: bool = True,
    save_hist: bool = False,
    return_df: bool = False,
    dry_run: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Pipeline completo de predicciones V2: etapa, subetapa, target real y SHAP.

    Args:
        save_to_oracle: Si True, hace upsert en PMAT_PREDICTION.
        save_hist:      Si True, también inserta en PMAT_PREDICTION_HIST.
        return_df:      Si True, devuelve el DataFrame combinado.
        dry_run:        Si True, ejecuta todo pero no escribe en Oracle.

    Returns:
        DataFrame combinado si return_df=True, None en caso contrario.
    """
    try:
        from explainer import generar_explicacion  # noqa: PLC0415
    except ImportError:
        generar_explicacion = None  # type: ignore[assignment]
        logger.warning("explainer.py no disponible — SHAP desactivado.")

    fecha_ejecucion = datetime.now()
    logger.info(f"Iniciando predicciones V2 — {fecha_ejecucion.isoformat()}")

    df_raw = load_dataset_limpio()

    resultados = []
    for tipo in TIPOS:
        logger.info(f"--- Segmento: {tipo} ---")

        model = cargar_modelo(tipo)
        model_features = [f for f in model.feature_names_in_ if f != "target"]

        df_model, _, df_ids_base = preprocess(df_raw, tipo, model_features=model_features)

        if df_model.empty:
            logger.warning(f"No hay filas para '{tipo}' — omitiendo.")
            continue

        # ETAPA y SUBETAPA ya vienen en df_ids_base desde preprocessor.preprocess()
        # (capturadas en el paso 5 de preprocess, antes de eliminar COLS_ID)
        df_ids_completo = df_ids_base.copy()

        preds = predecir(model, df_model)

        # Generar explicaciones SHAP
        explicaciones = None
        if generar_explicacion is not None:
            try:
                explicaciones = generar_explicacion(model, df_model)
            except Exception as e:
                logger.warning(f"SHAP falló para '{tipo}': {e} — continuando sin explicaciones.")

        df_resultado = construir_resultado_v2(
            df_ids_completo, preds, tipo, fecha_ejecucion, explicaciones
        )
        resultados.append(df_resultado)

        n_mat = int((df_resultado["TARGET_PRED"] == 1).sum())
        logger.info(
            f"Segmento '{tipo}': {len(df_resultado)} predicciones | "
            f"Matrícula prevista: {n_mat} ({n_mat/len(df_resultado)*100:.1f}%)"
        )

    if not resultados:
        logger.error("No se generaron predicciones para ningún segmento.")
        return None

    df_final = pd.concat(resultados, ignore_index=True)
    logger.info(f"Total predicciones V2: {len(df_final)}")

    if save_to_oracle and not dry_run:
        guardar_en_oracle_v2(df_final, save_hist=save_hist)
    elif dry_run:
        logger.info("[DRY-RUN] No se escriben datos en Oracle.")

    if return_df:
        return df_final

    return None


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Predictor de matrícula UNAV")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Usa run_predictions() (INSERT histórico, tabla PREDICCIONES)",
    )
    parser.add_argument(
        "--save-hist",
        action="store_true",
        help="Guarda también en PMAT_PREDICTION_HIST (solo con modo V2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta sin escribir en Oracle",
    )
    args = parser.parse_args()

    if args.legacy:
        run_predictions(save_to_oracle=True)
    else:
        run_predictions_v2(
            save_to_oracle=True,
            save_hist=args.save_hist,
            dry_run=args.dry_run,
        )
