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

import logging
import pandas as pd
import numpy as np
from typing import Literal

logger = logging.getLogger(__name__)

# ─── Constantes de negocio ────────────────────────────────────────────────────

COLS_ID = ["ACCOUNTID", "ID", "ID18__PC", "BIRTHDATE", "CreatedDate"]

VARS_EXCLUIR = [
    "desmatriculado", "MINIMUMPAYMENTPAYED", "CH_PAGO_SUPERIOR",
    "PL_Etapa__c", "PL_Subetapa__c", "ACC_DTT_FECHAULTIMAACTIVIDAD",
    "NAMEX", "YEARPERSONBIRTHDATE", "PAID_AMOUNT",
    "PC1", "PC2", "CreatedDate", "cluster", "interpretacion_cluster",
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
    return df


def split_grado_master(
    df: pd.DataFrame,
    tipo: Literal["grado", "master"],
) -> pd.DataFrame:
    """
    Filtra el dataset por tipo de titulación.

    Args:
        df:   DataFrame completo con columna TITULACION.
        tipo: 'grado' excluye filas con 'MASTER', 'master' las incluye solo.

    Returns:
        DataFrame filtrado.
    """
    if "TITULACION" not in df.columns:
        raise ValueError("El DataFrame no contiene la columna TITULACION")

    mask_master = df["TITULACION"].str.contains("MASTER", case=False, na=False)

    if tipo == "grado":
        result = df[~mask_master].copy()
    else:
        result = df[mask_master].copy()

    logger.info(f"Split '{tipo}': {len(result)} filas (de {len(df)} totales)")
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

    object_cols = set(df.select_dtypes(include=["object", "str"]).columns)
    const_cols = set(df.columns[df.nunique() <= 1])
    pca_cols = {c for c in df.columns if c.upper().startswith("PC")}

    safe = [
        c for c in df.columns
        if c not in excluir | object_cols | const_cols | pca_cols
    ]

    logger.debug(f"safe_cols: {len(safe)} columnas seleccionadas")
    return safe


def preprocess(
    df: pd.DataFrame,
    tipo: Literal["grado", "master"],
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """
    Pipeline completo de preprocesado pre-PyCaret.

    Orquesta split → etapa ordinal → imputación → vinculación →
    drop IDs → selección safe_cols.

    Args:
        df:   DATASET_LIMPIO completo (tal como sale de Oracle).
        tipo: 'grado' o 'master'.

    Returns:
        df_model   — DataFrame con safe_cols listo para predict_model()
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

    # 5. Guardar IDs y target para auditoría antes de eliminarlos
    cols_auditoria = [c for c in ["ID"] + ([TARGET] if TARGET in df_seg.columns else [])
                      if c in df_seg.columns]
    df_ids = df_seg[cols_auditoria].reset_index(drop=True)

    # 6. Drop columnas identificativas
    df_seg = df_seg.drop(columns=[c for c in COLS_ID if c in df_seg.columns])

    # 7. Drop target del DataFrame de features
    if TARGET in df_seg.columns:
        df_seg = df_seg.drop(columns=[TARGET])

    # 8. Selección de safe_cols
    safe_cols = get_safe_cols(df_seg, target_col=TARGET)

    df_model = df_seg[safe_cols].reset_index(drop=True)

    logger.info(
        f"Preprocesado '{tipo}' completado: "
        f"{len(df_model)} filas × {len(safe_cols)} features"
    )
    return df_model, safe_cols, df_ids
