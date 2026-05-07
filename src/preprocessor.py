"""
src/preprocessor.py
===================
Fase 4 · Preprocesado pre-PyCaret para predicciones.

Replica exactamente la lógica de preprocesado del notebook 03_Modelado.ipynb
antes de llamar a predict_model(). PyCaret maneja internamente la normalización;
aquí se aplican las transformaciones de negocio:

    1. Carga de DATASET_LIMPIO desde Oracle
    2. Separación Grado / Máster por TITULACION
    3. Construcción de etapa_compuesta + etapa_ordinal_num (automática)
    4. Imputación lógica (0 para notas/importes, media para renta, 0 para beca)
    5. Feature vinculacion_previa (max de 6 indicadores de alumno previo)
    6. Drop de columnas identificativas
    7. Selección de safe_cols (sin object, sin constantes, sin PCA, sin vars_excluir)

Devuelve:
    df_model   — DataFrame listo para predict_model() con safe_cols
    safe_cols  — lista de columnas usadas como features
    df_ids     — DataFrame completo con ID y target (si existe) para auditoría
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from typing import Literal

logger = logging.getLogger(__name__)

# ─── Mapeo Oracle (mayúsculas) → nombres que esperan los modelos ──────────────
# Las columnas de DATASET_LIMPIO en Oracle están en MAYÚSCULAS o con sufijos
# distintos a los del CSV original con el que se entrenaron los modelos.

ORACLE_COLUMN_MAP = {
    # Columnas que Oracle guarda en mayúsculas pero el modelo espera en minúsculas
    "CREATEDDATE":             "CreatedDate",
    "PL_ETAPA__C":             "PL_Etapa__c",
    "PL_SUBETAPA__C":          "PL_Subetapa__c",
    "FO_RENTAFAM_GES__C":      "FO_rentaFam_ges__c",
    "CU_PRECIOORDINARIO_DEF__C": "CU_precioOrdinario_def__c",
    "CU_PRECIOAPLICADO_DEF__C":  "CU_precioAplicado_def__c",
    "TIEMPO_ETAPA_DIAS":       "tiempo_etapa_dias",
    "TIEMPO_ENTRE_ETAPAS_DIAS": "tiempo_entre_etapas_dias",
    "NUM_ASISTENCIAS_ACUM":    "num_asistencias_acum",
    "NUM_SOLICITUDES_ACUM":    "num_solicitudes_acum",
    # Columna renombrada entre CSV y Oracle
    "PORCENTAJE_PAGADO_FINAL": "PAID_PERCENT",
    # target en Oracle en mayúsculas
    "TARGET":                  "target",
    "DESMATRICULADO":          "desmatriculado",
}

# Columnas que no existen en Oracle pero los modelos necesitan → se añaden con 0
# CU_IMPORTE_TOTAL y NU_MEDIA_EXPEDIENTE_UNIVERSITA se extraen de SF
# (Opportunity y Account respectivamente) y llegan a DATASET_LIMPIO con datos reales
COLS_ADD_ZERO: list = []

# ─── Constantes de negocio ────────────────────────────────────────────────────

COLS_ID = ["ACCOUNTID", "ID", "ID18__PC", "BIRTHDATE", "CreatedDate"]

VARS_EXCLUIR = [
    # Nombres originales del notebook
    "desmatriculado", "MINIMUMPAYMENTPAYED", "CH_PAGO_SUPERIOR",
    "PL_Etapa__c", "PL_Subetapa__c", "ACC_DTT_FECHAULTIMAACTIVIDAD",
    "NAMEX", "YEARPERSONBIRTHDATE", "PAID_AMOUNT",
    "PC1", "PC2", "CreatedDate", "cluster", "interpretacion_cluster",
    # Nombres adicionales que Oracle puede tener y no se deben usar como features
    "MINIMUMPAYMENTPAYED", "CH_PAGO_SUPERIOR", "FUENTE_DATOS",
    "PAID_AMOUNT", "YEARPERSONBIRTHDATE",
    # Clasificadores de tipo de solicitud — solo para split, nunca como feature
    "RECORDTYPEID", "RECORDTYPENAME",
]

VARS_CERO_LOGICO = [
    "NU_NOTA_MEDIA_ADMISION",
    "NU_NOTA_MEDIA_1_BACH__PC",
    "NU_RESULTADO_ADMISION_PUNTOS",
    "CU_IMPORTE_TOTAL",
    "CU_precioOrdinario_def__c",
    "CU_precioAplicado_def__c",
    "PAID_PERCENT",
    "NU_MEDIA_EXPEDIENTE_UNIVERSITA",
]

COLS_VINCULACION = [
    "CH_ALUMNO__PC", "CH_ESTUDIANTE__PC", "CH_ANTIGUO_ALUMNO__PC",
    "CH_ALUMNI__PC", "CH_ANTIGUOALUMNO_INTERCAMBIO",
    "CH_HIJO_ANTIGUO_ALUMNO__PC",
]

TARGET = "target"


# ─── Funciones públicas ───────────────────────────────────────────────────────

def normalizar_columnas_oracle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adapta las columnas del DATASET_LIMPIO de Oracle al esquema que esperan
    los modelos entrenados (nombres del notebook original).

    Pasos:
    1. Renombra las columnas según ORACLE_COLUMN_MAP
    2. Añade con valor 0 las columnas que faltan en Oracle pero que el modelo necesita
    """
    df = df.rename(columns=ORACLE_COLUMN_MAP)

    for col in COLS_ADD_ZERO:
        if col not in df.columns:
            df[col] = 0.0
            logger.debug(f"Columna '{col}' no encontrada en Oracle — añadida con 0")

    return df


def load_dataset_limpio() -> pd.DataFrame:
    """
    Carga la tabla DATASET_LIMPIO desde Oracle y la devuelve como DataFrame.

    Usa OracleConnector del proyecto (src/oracle_connector.py).
    Las credenciales se leen del .env en la raíz del proyecto.
    """
    from oracle_connector import OracleConnector

    logger.info("Cargando DATASET_LIMPIO desde Oracle...")
    conn = OracleConnector()
    records = conn.read_table("DATASET_LIMPIO")
    df = pd.DataFrame(records)
    logger.info(f"DATASET_LIMPIO cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    df = normalizar_columnas_oracle(df)
    logger.info("Columnas normalizadas (Oracle → modelo)")
    return df


# IDs de RecordType de Salesforce — fuente: equipo UNAV (Marta, abr-2026)
_RECORDTYPE_GRADO  = {"012w0000000K4QPAA0", "012w0000000K4QTAA0"}
_RECORDTYPE_MASTER = {"012w0000000K4QUAA0", "012w0000000K4QQAA0"}


def split_grado_master(
    df: pd.DataFrame,
    tipo: Literal["grado", "master"],
) -> pd.DataFrame:
    """
    Filtra el dataset por tipo de titulación.

    Prioridad de columna clasificadora:
      1. RECORDTYPEID  — IDs exactos de RecordType de Salesforce (más fiable).
      2. RECORDTYPENAME — subcadena "ster" (cubre "Master" y "Máster" con tilde).
      3. TITULACION     — subcadena "MASTER" (fallback para datos históricos Excel).

    Args:
        df:   DataFrame con al menos una de las columnas anteriores.
        tipo: 'grado' o 'master'.

    Returns:
        DataFrame filtrado.

    Raises:
        ValueError: Si no se encuentra ninguna columna clasificadora.
    """
    if "RECORDTYPEID" in df.columns:
        ids_tipo = _RECORDTYPE_GRADO if tipo == "grado" else _RECORDTYPE_MASTER
        result = df[df["RECORDTYPEID"].isin(ids_tipo)].copy()
        n_master = int(df["RECORDTYPEID"].isin(_RECORDTYPE_MASTER).sum())
        col_usada = "RECORDTYPEID"
    elif "RECORDTYPENAME" in df.columns:
        mask_master = df["RECORDTYPENAME"].str.contains("ster", case=False, na=False)
        result = df[~mask_master if tipo == "grado" else mask_master].copy()
        n_master = int(mask_master.sum())
        col_usada = "RECORDTYPENAME"
    elif "TITULACION" in df.columns:
        mask_master = df["TITULACION"].str.contains("MASTER", case=False, na=False)
        result = df[~mask_master if tipo == "grado" else mask_master].copy()
        n_master = int(mask_master.sum())
        col_usada = "TITULACION"
    else:
        raise ValueError(
            "El DataFrame no contiene RECORDTYPEID, RECORDTYPENAME ni TITULACION — "
            "no es posible separar grado de máster."
        )

    logger.info(
        "Split '%s' (col=%s): %d filas (de %d totales | máster en dataset=%d)",
        tipo, col_usada, len(result), len(df), n_master,
    )
    if tipo == "master" and len(result) == 0:
        logger.warning(
            "Segmento 'master' vacío — revisar valores de %s en DATASET_LIMPIO.",
            col_usada,
        )
    return result


def calcular_orden_automatico(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Calcula el orden de etapas automáticamente por secuencia temporal media.

    Replica exactamente la función calcular_orden_automatico del notebook
    03_Modelado.ipynb. Crea dos columnas nuevas:
        - etapa_compuesta    : PL_Etapa__c + '__' + PL_Subetapa__c
        - etapa_ordinal_num  : posición ordinal basada en tiempo de aparición

    Args:
        df: DataFrame con columnas CreatedDate, PL_Etapa__c, PL_Subetapa__c, ID.

    Returns:
        (df modificado, diccionario de mapeo etapa_compuesta → orden)
    """
    df = df.copy()

    # Crear etapa compuesta
    df["CreatedDate"] = pd.to_datetime(df["CreatedDate"], errors="coerce")
    df["etapa_compuesta"] = (
        df["PL_Etapa__c"].fillna("NA").astype(str)
        + "__"
        + df["PL_Subetapa__c"].fillna("NA").astype(str)
    )

    # Primera aparición de cada etapa por oportunidad
    primeras = (
        df.groupby(["ID", "etapa_compuesta"])["CreatedDate"]
        .min()
        .reset_index()
    )

    # Ranking interno por oportunidad
    primeras["ranking_interno"] = (
        primeras.groupby("ID")["CreatedDate"].rank(method="first")
    )

    # Posición media global → orden automático
    orden_logico = (
        primeras.groupby("etapa_compuesta")["ranking_interno"]
        .mean()
        .sort_values()
        .reset_index()
    )
    orden_logico["orden_automatico"] = range(len(orden_logico))
    mapa_orden = dict(
        zip(orden_logico["etapa_compuesta"], orden_logico["orden_automatico"])
    )

    df["etapa_ordinal_num"] = df["etapa_compuesta"].map(mapa_orden)

    logger.debug(f"Etapas detectadas: {len(mapa_orden)}")
    return df, mapa_orden


def imputar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica imputación lógica de valores nulos.

    - VARS_CERO_LOGICO      → 0 (sin nota/importe = 0 tiene sentido de negocio)
    - FO_rentaFam_ges__c    → media del segmento
    - CH_MATRICULA_SUJETA_BECA → 0 (sin beca)

    Args:
        df: DataFrame del segmento (Grado o Máster).

    Returns:
        DataFrame con nulos imputados.
    """
    df = df.copy()

    for col in VARS_CERO_LOGICO:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if "FO_rentaFam_ges__c" in df.columns:
        media_renta = df["FO_rentaFam_ges__c"].mean()
        df["FO_rentaFam_ges__c"] = df["FO_rentaFam_ges__c"].fillna(media_renta)

    if "CH_MATRICULA_SUJETA_BECA" in df.columns:
        df["CH_MATRICULA_SUJETA_BECA"] = df["CH_MATRICULA_SUJETA_BECA"].fillna(0)

    return df


def crear_vinculacion_previa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la feature 'vinculacion_previa' como max de indicadores de alumno previo.

    Valor 1 si alguno de los indicadores es True/1, 0 en caso contrario.

    Args:
        df: DataFrame con columnas de vinculación (pueden no existir todas).

    Returns:
        DataFrame con columna 'vinculacion_previa' añadida.
    """
    df = df.copy()
    existentes = [c for c in COLS_VINCULACION if c in df.columns]

    if existentes:
        df["vinculacion_previa"] = (
            df[existentes].fillna(False).astype(int).max(axis=1)
        )
    else:
        logger.warning("No se encontraron columnas de vinculación previa; se asigna 0")
        df["vinculacion_previa"] = 0

    return df


def get_safe_cols(df: pd.DataFrame, target_col: str = TARGET) -> list[str]:
    """
    Devuelve la lista de columnas seguras para el modelo:
    excluye object, constantes, columnas PCA y VARS_EXCLUIR.

    Args:
        df:         DataFrame de entrenamiento/inferencia.
        target_col: Nombre de la columna target (excluida de features).

    Returns:
        Lista de nombres de columnas válidas para el modelo.
    """
    excluir = set(VARS_EXCLUIR + COLS_ID + [target_col])

    object_cols = set(df.select_dtypes(include=["object"]).columns)
    const_cols = set(df.columns[df.nunique() <= 1])
    pca_cols = {c for c in df.columns if c.upper().startswith("PC")}

    safe = [
        c for c in df.columns
        if c not in excluir | object_cols | const_cols | pca_cols
    ]

    logger.debug(f"safe_cols: {len(safe)} columnas seleccionadas")
    return safe


def preparar_features_modelo(df: pd.DataFrame, model_features: list[str]) -> pd.DataFrame:
    """
    Prepara exactamente las columnas que el modelo espera para inferencia.

    A diferencia de get_safe_cols (diseñada para entrenamiento), esta función
    NO filtra constantes ni columnas object — simplemente:
    1. Selecciona las features que el modelo necesita
    2. Añade con 0 las que faltan en el DataFrame
    3. Convierte columnas object a numérico (pd.to_numeric con coerce)

    Args:
        df:             DataFrame preprocesado (sin ID ni target).
        model_features: Lista de features que espera el modelo (model.feature_names_in_).

    Returns:
        DataFrame con exactamente las columnas en model_features, en ese orden.
    """
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]
    for col in model_features:
        if col not in df.columns:
            df[col] = 0.0
            logger.debug(f"Feature '{col}' ausente — añadida con 0")
        elif df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            logger.debug(f"Feature '{col}' convertida de object a numérico")
    return df[model_features]


def preprocess(
    df: pd.DataFrame,
    tipo: Literal["grado", "master"],
    model_features: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """
    Pipeline completo de preprocesado pre-PyCaret.

    Orquesta split → etapa ordinal → imputación → vinculación →
    drop IDs → preparación de features para el modelo.

    Args:
        df:             DATASET_LIMPIO completo (tal como sale de Oracle).
        tipo:           'grado' o 'master'.
        model_features: Features exactas del modelo (model.feature_names_in_ sin 'target').
                        Si se proporciona, se usa preparar_features_modelo() en lugar de
                        get_safe_cols(), garantizando compatibilidad total con el modelo.

    Returns:
        df_model   — DataFrame listo para predict_model()
        safe_cols  — lista de columnas usadas como features
        df_ids     — DataFrame con ID (y target si existe) para auditoría
    """
    logger.info(f"Iniciando preprocesado para '{tipo}'...")

    # 1. Split por tipo de titulación
    df_seg = split_grado_master(df, tipo)

    # 2. Etapa ordinal automática
    df_seg, _ = calcular_orden_automatico(df_seg)

    # 3. Imputación lógica
    df_seg = imputar(df_seg)

    # 4. Vinculación previa
    df_seg = crear_vinculacion_previa(df_seg)

    # 5. Guardar IDs, target y etapa/subetapa para auditoría antes de eliminarlos
    COLS_AUDITORIA_EXTRA = ["PL_Etapa__c", "PL_Subetapa__c", "CreatedDate"]
    cols_auditoria = [
        c for c in ["ID"] + ([TARGET] if TARGET in df_seg.columns else []) + COLS_AUDITORIA_EXTRA
        if c in df_seg.columns
    ]
    df_ids = df_seg[cols_auditoria].reset_index(drop=True)

    # 6. Drop columnas identificativas
    df_seg = df_seg.drop(columns=[c for c in COLS_ID if c in df_seg.columns])

    # 7. Drop target del DataFrame de features
    if TARGET in df_seg.columns:
        df_seg = df_seg.drop(columns=[TARGET])

    # 8. Selección de columnas
    if model_features is not None:
        # Modo inferencia: usar exactamente las features del modelo
        df_model = preparar_features_modelo(df_seg, model_features).reset_index(drop=True)
        safe_cols = model_features
    else:
        # Modo exploración: filtro conservador para análisis
        safe_cols = get_safe_cols(df_seg, target_col=TARGET)
        df_model = df_seg[safe_cols].reset_index(drop=True)

    logger.info(
        f"Preprocesado '{tipo}' completado: "
        f"{len(df_model)} filas × {len(safe_cols)} features"
    )
    return df_model, safe_cols, df_ids
