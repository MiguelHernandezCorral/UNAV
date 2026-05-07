"""
cleaner.py
==========
FASE 2 – Limpieza y construcción del dataset de modelado
==========================================================
Replica la lógica del notebook 01_Limpieza copy.ipynb pero leyendo
directamente de Oracle (sin Excel intermedios) y guardando el resultado
también en Oracle (tabla DATASET_LIMPIO).

Pasos:
  1.  Carga de tablas desde Oracle
  2.  Renombrado de columnas (Oracle → convención notebook)
  3.  Validación de campos: notebook vs Oracle (comparativa)
  4.  Eliminación de columnas con >90 % de NA
  5.  Creación del target (matricula formalizada / desmatriculado)
  6.  Join Opportunity × Account
  7.  Variables derivadas (titulación, centro, año nacimiento, tipo solicitud)
  8.  Normalización del plazo de admisión
  9.  Join con ECB + porcentaje pagado
  10. Tiempos en cada etapa (calcular_tiempos_etapas)
  11. Control de leakage por hitos temporales (limpiar_historial_por_hitos)
  12. Integración de actividades acumuladas
  13. Detección y corrección de leakage de pago
  14. Eliminación de registros sin target
  15. Selección de columnas finales
  16. Guardado en Oracle + documentación Markdown

Estado de fases:
  [DONE] Fase 1 – Ingesta SF → Oracle
  [DONE] Fase 2 – Limpieza y construcción del dataset

Uso:
    python src/cleaner.py
    python src/cleaner.py --recreate   # elimina DATASET_LIMPIO antes de insertar
"""

from __future__ import annotations

import sys
import logging
import logging.handlers
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from oracle_connector import OracleConnector

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
LOG_DIR  = BASE_DIR / "logs"
DOCS_DIR = BASE_DIR / "docs"
LOG_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

_root = logging.getLogger()
_root.setLevel(logging.INFO)
_fmt  = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s – %(message)s")

_console = logging.StreamHandler()
_console.setFormatter(_fmt)
_root.addHandler(_console)

_file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_DIR / "cleaner.log", when="midnight", backupCount=7, encoding="utf-8"
)
_file_handler.setFormatter(_fmt)
_root.addHandler(_file_handler)

logger = logging.getLogger(__name__)


# ===========================================================================
# MAPEO DE COLUMNAS: Oracle (uppercase) → Convención del notebook
# ===========================================================================
# Permite comparar qué campos del notebook están en Oracle y con qué nombre.
# El renombrado se aplica al cargar los DataFrames para que el resto del
# código de limpieza sea idéntico al notebook.
# ===========================================================================

OPP_RENAME: dict[str, str] = {
    "PL_CURSO_ACADEMICO__C":                  "PL_CURSO_ACADEMICO",
    "CH_NACIONAL__C":                          "CH_NACIONAL",
    "NU_NOTA_MEDIA_ADMISION__C":               "NU_NOTA_MEDIA_ADMISION",
    "CH_PRUEBAS_CALIFICADAS__C":               "CH_PRUEBAS_CALIFICADAS",
    "NU_RESULTADO_ADMISION_PUNTOS__C":         "NU_RESULTADO_ADMISION_PUNTOS",
    "PL_RESOLUCION_DEFINITIVA__C":             "PL_RESOLUCION_DEFINITIVA",
    "MINIMUMPAYMENTPAYED__C":                  "MINIMUMPAYMENTPAYED",
    "PAID_AMOUNT__C":                          "PAID_AMOUNT",
    "PAID_PERCENT__C":                         "PAID_PERCENT",
    "CH_PAGO_SUPERIOR__C":                     "CH_PAGO_SUPERIOR",
    "CH_MATRICULA_SUJETA_BECA__C":             "CH_MATRICULA_SUJETA_BECA",
    "CU_IMPORTE_TOTAL__C":                     "CU_IMPORTE_TOTAL",
    "NU_PREFERENCIA__C":                       "NU_PREFERENCIA",
    "PL_ORIGEN_DE_SOLICITUD__C":               "PL_ORIGEN_DE_SOLICITUD",
    "PL_PLAZO_ADMISION__C":                    "PL_PLAZO_ADMISION",
    "RECORDTYPE.NAME":                         "RECORDTYPENAME",
    "PL_SUBETAPA__C":                          "PL_SUBETAPA",
    # Columnas derivadas de relaciones anidadas
    "LK_TITULACION_DEFINITIVA__R.NAME":        "TITULACION",
    "LK_CENTROENSENANZA__R.NAME":              "CENTROENSENANZA",
}

ACC_RENAME: dict[str, str] = {
    "CH_AYUDA_FINANCIACION__C":               "CH_AYUDA_FINANCIACION",
    "ACC_DTT_FECHAULTIMAACTIVIDAD__C":         "ACC_DTT_FECHAULTIMAACTIVIDAD",
    "PL_SITUACION_SOCIO_ECONOMICA__PC":        "PL_SITUACION_SOCIO_ECONOMICA",
    # Nombre diferente: Oracle usa ANTIGUOALUMNO, notebook CH_ANTIGUO_ALUMNO
    "CH_ANTIGUOALUMNO__PC":                   "CH_ANTIGUO_ALUMNO__PC",
    # Truncado en el notebook (máx 30 chars de Oracle)
    "CH_HERMANOS_ESTUDIANDO_UNAV__PC":        "CH_HERMANOS_ESTUDIANDO_UNAV__P",
    "CH_ANTIGUOALUMNO_INTERCAMBIO__PC":       "CH_ANTIGUOALUMNO_INTERCAMBIO",
    "NU_MEDIA_EXPEDIENTE_UNIVERSITARIO__C":   "NU_MEDIA_EXPEDIENTE_UNIVERSITA",
}

ECBS_RENAME: dict[str, str] = {
    "LK_OPORTUNIDAD__C":          "LK_oportunidad__c",
    "FO_RENTAFAM_GES__C":         "FO_rentaFam_ges__c",
    "CU_PRECIOORDINARIO_DEF__C":  "CU_precioOrdinario_def__c",
    "CU_PRECIOAPLICADO_DEF__C":   "CU_precioAplicado_def__c",
}

STAGE_RENAME: dict[str, str] = {
    "PL_ETAPA__C":          "PL_Etapa__c",
    "PL_SUBETAPA__C":       "PL_Subetapa__c",
    "LK_OPORTUNIDAD__C":    "LK_Oportunidad__c",
    "CREATEDDATE":          "CreatedDate",
    "FECHA_FIN_ETAPA__C":   "Fecha_fin_etapa__c",
}

ACT_RENAME: dict[str, str] = {
    "CONTACTID":                                        "ContactId",
    "CREATEDDATE":                                      "CreatedDate",
    "ESTADO_DEL_MIEMBRO__C":                            "Estado_del_miembro__c",
    "CAMPAIGN.ACADEMICCOURSE__C":                       "Campaign.AcademicCourse__c",
    "CAMPAIGN.LK_TIPOACTIVIDADPROMOCION__R.NAME":       "Campaign.LK_tipoActividadPromocion__r.Name",
}

# ---------------------------------------------------------------------------
# Renames para tablas HIST (cargadas desde el Excel histórico)
# Las columnas del Excel ya están en formato "notebook"; sólo se necesitan
# ajustes puntuales respecto a las tablas Oracle API.
# ---------------------------------------------------------------------------
# OPPORTUNITY_HIST: TITULACION = inicial, TITULACION_DEF = definitiva.
# Renombramos para que "TITULACION" sea siempre la definitiva (consistente con API).
OPP_HIST_RENAME: dict[str, str] = {
    "TITULACION":     "TITULACION_INICIAL",  # renombrar la inicial para liberar el nombre
    "TITULACION_DEF": "TITULACION",          # la definitiva toma el nombre estándar
}
# ECBS_HIST, STAGE_HISTORY_HIST, ACTIVITY_HIST usan los mismos renames que las
# tablas Oracle API (los valores en Oracle ya están en mayúsculas).
ECBS_HIST_RENAME  = ECBS_RENAME
STAGE_HIST_RENAME = STAGE_RENAME
ACT_HIST_RENAME   = ACT_RENAME


# ===========================================================================
# CAMPOS ESPERADOS POR EL NOTEBOOK (columnas_seleccionadas finales)
# Mapeados a: tabla Oracle, columna Oracle, estado
# ===========================================================================

# Formato: notebook_name → (tabla_oracle, columna_oracle, notas)
NOTEBOOK_FIELDS: dict[str, tuple[str, str | None, str]] = {
    # Identificadores
    "ACCOUNTID":                    ("OPPORTUNITY", "ACCOUNTID",                       "directo"),
    "ID":                           ("OPPORTUNITY", "ID",                              "directo"),
    "ID18__PC":                     ("ACCOUNT",     "ID18__PC",                        "directo"),
    # Target
    "target":                       ("COMPUTED",    None,                              "derivado: Matrícula OOGG + Formalizada"),
    "desmatriculado":               ("COMPUTED",    None,                              "derivado: PL_Subetapa__c=Desmatriculado"),
    # Oportunidad
    "PL_CURSO_ACADEMICO":           ("OPPORTUNITY", "PL_CURSO_ACADEMICO__C",           "renombrado"),
    "CH_NACIONAL":                  ("OPPORTUNITY", "CH_NACIONAL__C",                  "renombrado"),
    "NU_NOTA_MEDIA_ADMISION":       ("OPPORTUNITY", "NU_NOTA_MEDIA_ADMISION__C",        "renombrado"),
    "NU_NOTA_MEDIA_1_BACH__PC":     ("ACCOUNT",     "NU_NOTA_MEDIA_1_BACH__PC",        "directo"),
    "CH_PRUEBAS_CALIFICADAS":       ("OPPORTUNITY", "CH_PRUEBAS_CALIFICADAS__C",        "renombrado"),
    "NU_RESULTADO_ADMISION_PUNTOS": ("OPPORTUNITY", "NU_RESULTADO_ADMISION_PUNTOS__C", "renombrado"),
    "PL_RESOLUCION_DEFINITIVA":     ("OPPORTUNITY", "PL_RESOLUCION_DEFINITIVA__C",      "renombrado"),
    "TITULACION":                   ("OPPORTUNITY", "LK_TITULACION_DEFINITIVA__R.NAME","renombrado (relación)"),
    "CENTROENSENANZA":              ("OPPORTUNITY", "LK_CENTROENSENANZA__R.NAME",       "renombrado (relación)"),
    "MINIMUMPAYMENTPAYED":          ("OPPORTUNITY", "MINIMUMPAYMENTPAYED__C",           "renombrado"),
    "PAID_AMOUNT":                  ("OPPORTUNITY", "PAID_AMOUNT__C",                   "renombrado"),
    "PAID_PERCENT":                 ("OPPORTUNITY", "PAID_PERCENT__C",                  "renombrado"),
    "CH_PAGO_SUPERIOR":             ("OPPORTUNITY", "CH_PAGO_SUPERIOR__C",              "renombrado"),
    "CH_MATRICULA_SUJETA_BECA":     ("OPPORTUNITY", "CH_MATRICULA_SUJETA_BECA__C",      "renombrado"),
    "CH_AYUDA_FINANCIACION":        ("ACCOUNT",     "CH_AYUDA_FINANCIACION__C",         "renombrado"),
    "CU_IMPORTE_TOTAL":             ("OPPORTUNITY", "CU_IMPORTE_TOTAL__C",              "renombrado"),
    "CH_VISITACAMPUS__PC":          ("ACCOUNT",     "CH_VISITACAMPUS__PC",              "directo"),
    "CH_ENTREVISTA_PERSONAL__PC":   ("ACCOUNT",     "CH_ENTREVISTA_PERSONAL__PC",       "directo"),
    "ACC_DTT_FECHAULTIMAACTIVIDAD": ("ACCOUNT",     "ACC_DTT_FECHAULTIMAACTIVIDAD__C",  "renombrado"),
    "NU_PREFERENCIA":               ("OPPORTUNITY", "NU_PREFERENCIA__C",                "renombrado"),
    "PL_Etapa__c":                  ("STAGE_HISTORY","PL_ETAPA__C",                    "renombrado"),
    "PL_Subetapa__c":               ("STAGE_HISTORY","PL_SUBETAPA__C",                 "renombrado"),
    "CH_HIJO_EMPLEADO__PC":         ("ACCOUNT",     "CH_HIJO_EMPLEADO__PC",             "directo"),
    "CH_HIJO_PROFESOR_ASOCIADO__PC":("ACCOUNT",     "CH_HIJO_PROFESOR_ASOCIADO__PC",    "directo"),
    "CH_HERMANOS_ESTUDIANDO_UNAV__P":("ACCOUNT",    "CH_HERMANOS_ESTUDIANDO_UNAV__PC",  "truncado en notebook"),
    "CH_HIJO_MEDICO__PC":           ("ACCOUNT",     "CH_HIJO_MEDICO__PC",               "directo"),
    "YEARPERSONBIRTHDATE":          ("ACCOUNT",     "PERSONBIRTHDATE",                  "derivado: año de PERSONBIRTHDATE"),
    "NAMEX":                        ("—",           None,                               "⚠️ NO existe en Oracle (no se seleccionó)"),
    "CH_FAMILIA_NUMEROSA__PC":      ("ACCOUNT",     "CH_FAMILIA_NUMEROSA__PC",          "directo"),
    "PL_SITUACION_SOCIO_ECONOMICA": ("ACCOUNT",     "PL_SITUACION_SOCIO_ECONOMICA__PC", "renombrado"),
    "LEADSOURCE":                   ("OPPORTUNITY", "LEADSOURCE",                       "directo"),
    "PL_ORIGEN_DE_SOLICITUD":       ("OPPORTUNITY", "PL_ORIGEN_DE_SOLICITUD__C",        "renombrado"),
    "PL_PLAZO_ADMISION":            ("OPPORTUNITY", "PL_PLAZO_ADMISION__C",             "renombrado"),
    "RECORDTYPENAME":               ("OPPORTUNITY", "RECORDTYPE.NAME",                  "renombrado (relación anidada)"),
    "PLAZO_ADMISION_LIMPIO":        ("COMPUTED",    None,                               "derivado: normalización PL_PLAZO_ADMISION"),
    "FO_rentaFam_ges__c":           ("ECBS",        "FO_RENTAFAM_GES__C",               "renombrado"),
    "CU_precioOrdinario_def__c":    ("ECBS",        "CU_PRECIOORDINARIO_DEF__C",         "renombrado"),
    "CU_precioAplicado_def__c":     ("ECBS",        "CU_PRECIOAPLICADO_DEF__C",          "renombrado"),
    "PORCENTAJE_PAGADO_FINAL":      ("COMPUTED",    None,                               "derivado: precio_aplicado/precio_ordinario*100"),
    "tiempo_etapa_dias":            ("COMPUTED",    None,                               "derivado: Fecha_fin - CreatedDate"),
    "tiempo_entre_etapas_dias":     ("COMPUTED",    None,                               "derivado: lag entre etapas consecutivas"),
    "num_asistencias_acum":         ("COMPUTED",    None,                               "derivado: actividades asistidas acumuladas"),
    "num_solicitudes_acum":         ("COMPUTED",    None,                               "derivado: solicitudes acumuladas"),
    "CH_ALUMNO__PC":                ("ACCOUNT",     "CH_ALUMNO__PC",                    "directo"),
    "CH_ESTUDIANTE__PC":            ("ACCOUNT",     "CH_ESTUDIANTE__PC",                "directo"),
    "CH_ANTIGUO_ALUMNO__PC":        ("ACCOUNT",     "CH_ANTIGUOALUMNO__PC",             "⚠️ nombre distinto: Oracle = CH_ANTIGUOALUMNO__PC"),
    "CH_ALUMNI__PC":                ("ACCOUNT",     "CH_ALUMNI__PC",                    "directo"),
    "CH_ANTIGUOALUMNO_INTERCAMBIO": ("ACCOUNT",     "CH_ANTIGUOALUMNO_INTERCAMBIO__PC", "renombrado"),
    "CH_HIJO_ANTIGUO_ALUMNO__PC":   ("ACCOUNT",     "CH_HIJO_ANTIGUO_ALUMNO__PC",       "directo"),
    "CreatedDate":                  ("STAGE_HISTORY","CREATEDDATE",                     "renombrado"),
    "NU_MEDIA_EXPEDIENTE_UNIVERSITA":("ACCOUNT",    "NU_MEDIA_EXPEDIENTE_UNIVERSITARIO__C","⚠️ truncado en notebook (30 chars)"),
}


# ===========================================================================
# UTILIDADES DE LIMPIEZA (equivalentes a utils.py del notebook)
# ===========================================================================

def _step(n: int, total: int, desc: str) -> None:
    logger.info("─" * 60)
    logger.info("PASO %d/%d │ %s", n, total, desc)
    logger.info("─" * 60)


COLS_NUNCA_ELIMINAR = [
    "CU_IMPORTE_TOTAL",                # Importe total oportunidad — feature modelo Grado/Master
    "NU_MEDIA_EXPEDIENTE_UNIVERSITA",  # Nota expediente universitario — feature modelo Master
]


def eliminar_columnas_na(
    df: pd.DataFrame,
    umbral: float = 90.0,
    proteger: list = COLS_NUNCA_ELIMINAR,
) -> pd.DataFrame:
    """Elimina columnas con más de umbral% de valores NA.
    Las columnas en `proteger` nunca se eliminan aunque superen el umbral.
    """
    pct = df.isna().mean() * 100
    cols_drop = [c for c in pct[pct > umbral].index.tolist() if c not in proteger]
    cols_protegidas = [c for c in proteger if c in df.columns and pct.get(c, 0) > umbral]
    if cols_drop:
        logger.info("  Columnas eliminadas (>%.0f%% NA): %d → %s", umbral, len(cols_drop), cols_drop)
    if cols_protegidas:
        logger.info("  Columnas protegidas (>%.0f%% NA pero requeridas por modelo): %s", umbral, cols_protegidas)
    return df.drop(columns=cols_drop)


def crear_target(oportunidad: pd.DataFrame, historial_etapas: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las columnas 'target' y 'desmatriculado' en oportunidad.
    - target=1 si tiene etapa 'Matrícula OOGG' con subetapa 'Formalizada'
      Y NO tiene subetapa 'Desmatriculado'.
    - desmatriculado=1 si tiene subetapa 'Desmatriculado'.
    """
    mat_form = set(historial_etapas[
        (historial_etapas["PL_Etapa__c"] == "Matrícula OOGG") &
        (historial_etapas["PL_Subetapa__c"] == "Formalizada")
    ]["LK_Oportunidad__c"].unique())

    desmat = set(historial_etapas[
        historial_etapas["PL_Subetapa__c"] == "Desmatriculado"
    ]["LK_Oportunidad__c"].unique())

    ids_target_1 = mat_form - desmat
    oportunidad["target"] = oportunidad["ID"].isin(ids_target_1).astype(int)
    oportunidad["desmatriculado"] = oportunidad["ID"].isin(desmat).astype(int)

    total_opps = oportunidad["ID"].nunique()
    logger.info("  Matrículas formalizadas: %d (%.1f%% de %d oportunidades)",
                len(mat_form), len(mat_form) / total_opps * 100 if total_opps else 0, total_opps)
    logger.info("  Desmatriculados: %d (%.1f%% de formalizados)",
                len(desmat), len(desmat) / len(mat_form) * 100 if mat_form else 0)
    logger.info("  Target=1: %d | Target=0: %d",
                oportunidad["target"].sum(), (oportunidad["target"] == 0).sum())
    return oportunidad


def calcular_tiempos_etapas(historial: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula tiempo en cada etapa y tiempo de transición entre etapas.
    Columnas añadidas: tiempo_etapa_dias, tiempo_entre_etapas_dias.
    """
    df = historial.copy()
    df["CreatedDate"]        = pd.to_datetime(df["CreatedDate"], errors="coerce")
    df["Fecha_fin_etapa__c"] = pd.to_datetime(df["Fecha_fin_etapa__c"], errors="coerce")
    df = df.sort_values(["LK_Oportunidad__c", "CreatedDate"])

    df["tiempo_etapa_dias"] = (
        df["Fecha_fin_etapa__c"] - df["CreatedDate"]
    ).dt.days.fillna(0).astype(int)

    siguiente = df.groupby("LK_Oportunidad__c")["CreatedDate"].shift(-1)
    df["tiempo_entre_etapas_dias"] = (
        siguiente - df["CreatedDate"]
    ).dt.days.fillna(0).astype(int)

    return df


def limpiar_historial_por_hitos(
    df_historial: pd.DataFrame,
    df_principal: pd.DataFrame
) -> pd.DataFrame:
    """
    Evita leakage temporal poniendo a NaN campos académicos/económicos
    si la fila del historial es ANTERIOR al hito que los desbloquea.

    Hito académico: etapa='Pruebas de admisión', subetapa='Pruebas calificadas'
    Hito económico: etapa='Matrícula admisión', subetapa='Pago mínimo'
    """
    df = df_historial.copy()
    # Drop stage_history's own record ID to avoid column name collision with
    # opportunity ID during the merge below (both are named "ID")
    df = df.drop(columns=["ID"], errors="ignore")
    df["CreatedDate"] = pd.to_datetime(df["CreatedDate"], errors="coerce")

    # Fecha del hito académico por oportunidad
    hito_acad = (
        df[(df["PL_Etapa__c"] == "Pruebas de admisión") &
           (df["PL_Subetapa__c"] == "Pruebas calificadas")]
        .groupby("LK_Oportunidad__c")["CreatedDate"].min()
        .reset_index().rename(columns={"CreatedDate": "fecha_pruebas_calificadas"})
    )

    # Fecha del hito económico por oportunidad
    hito_econ = (
        df[(df["PL_Etapa__c"] == "Matrícula admisión") &
           (df["PL_Subetapa__c"] == "Pago mínimo")]
        .groupby("LK_Oportunidad__c")["CreatedDate"].min()
        .reset_index().rename(columns={"CreatedDate": "fecha_matricula_iniciada"})
    )

    df = df.merge(hito_acad, on="LK_Oportunidad__c", how="left")
    df = df.merge(hito_econ, on="LK_Oportunidad__c", how="left")

    # Eliminar cols duplicadas del df_principal antes del merge
    cols_excluir = [c for c in df_principal.columns if c in df.columns and c != "ID"]
    df_princ = df_principal.drop(columns=cols_excluir)

    df_final = df.merge(df_princ, left_on="LK_Oportunidad__c", right_on="ID", how="left")
    df_final = df_final.reset_index(drop=True)

    # Columnas con leakage académico
    cols_acad = [c for c in [
        "NU_NOTA_MEDIA_ADMISION", "CH_PRUEBAS_CALIFICADAS",
        "NU_RESULTADO_ADMISION_PUNTOS", "PL_RESOLUCION_DEFINITIVA"
    ] if c in df_final.columns]

    # Columnas con leakage económico
    cols_econ = [c for c in [
        "MINIMUMPAYMENTPAYED", "PAID_AMOUNT", "PAID_PERCENT", "CH_PAGO_SUPERIOR",
        "CH_MATRICULA_SUJETA_BECA", "CH_AYUDA_FINANCIACION", "CU_IMPORTE_TOTAL"
    ] if c in df_final.columns]

    mask_acad = (
        df_final["fecha_pruebas_calificadas"].isna() |
        (df_final["CreatedDate"] < df_final["fecha_pruebas_calificadas"])
    )
    df_final.loc[mask_acad, cols_acad] = np.nan
    logger.info("  Leakage académico: %d filas saneadas", mask_acad.sum())

    mask_econ = (
        df_final["fecha_matricula_iniciada"].isna() |
        (df_final["CreatedDate"] < df_final["fecha_matricula_iniciada"])
    )
    df_final.loc[mask_econ, cols_econ] = np.nan
    logger.info("  Leakage económico: %d filas saneadas", mask_econ.sum())

    return df_final


def integrar_actividades_progresivo(
    df_master: pd.DataFrame,
    df_actividades: pd.DataFrame
) -> pd.DataFrame:
    """
    Agrega actividades acumuladas (asistencias y solicitudes) por candidato
    y curso académico, controlando la progresión temporal para evitar leakage.
    """
    logger.info("  Filtrando actividades...")
    keywords_excluir = ["mail", "email", "whatsapp", "masivo", "comunicación", "envío"]
    patron = "|".join(keywords_excluir)
    col_tipo = "Campaign.LK_tipoActividadPromocion__r.Name"

    if col_tipo not in df_actividades.columns:
        logger.warning("  Columna '%s' no encontrada en actividades. Se omite integración.", col_tipo)
        df_master["num_asistencias_acum"] = 0
        df_master["num_solicitudes_acum"] = 0
        return df_master

    df_act = df_actividades[
        df_actividades[col_tipo].notna() &
        (df_actividades[col_tipo].str.strip() != "") &
        ~df_actividades[col_tipo].str.contains(patron, case=False, na=False)
    ].copy()

    df_act["ActivityDate"] = pd.to_datetime(df_act["CreatedDate"], errors="coerce")
    df_master["MasterDate"] = pd.to_datetime(df_master["CreatedDate"], errors="coerce")
    df_act["status_lower"] = df_act["Estado_del_miembro__c"].str.lower().fillna("")

    df_master = df_master.reset_index().rename(columns={"index": "_fila_id"})

    if "ID18__PC" not in df_master.columns or "PL_CURSO_ACADEMICO" not in df_master.columns:
        logger.warning("  Columnas ID18__PC o PL_CURSO_ACADEMICO no encontradas. Se omite integración.")
        df_master["num_asistencias_acum"] = 0
        df_master["num_solicitudes_acum"] = 0
        return df_master.drop(columns=["_fila_id", "MasterDate"], errors="ignore")

    logger.info("  Cruzando por ID18__PC y curso académico...")
    df_joined = pd.merge(
        df_master[["_fila_id", "ID18__PC", "MasterDate", "PL_CURSO_ACADEMICO"]],
        df_act[["ContactId", "ActivityDate", "status_lower", "Campaign.AcademicCourse__c"]],
        left_on=["ID18__PC", "PL_CURSO_ACADEMICO"],
        right_on=["ContactId", "Campaign.AcademicCourse__c"],
        how="inner"
    )

    df_joined = df_joined[df_joined["ActivityDate"] < df_joined["MasterDate"]]
    df_joined["es_asiste"]    = (df_joined["status_lower"] == "asiste").astype(int)
    df_joined["es_solicitado"] = df_joined["status_lower"].isin(
        ["solicitado", "solicita asistir"]
    ).astype(int)

    resumen = df_joined.groupby("_fila_id").agg(
        num_asistencias_acum=("es_asiste",    "sum"),
        num_solicitudes_acum=("es_solicitado","sum"),
    ).reset_index()

    df_final = df_master.merge(resumen, on="_fila_id", how="left")
    df_final[["num_asistencias_acum", "num_solicitudes_acum"]] = (
        df_final[["num_asistencias_acum", "num_solicitudes_acum"]].fillna(0).astype(int)
    )
    df_final = df_final.drop(columns=["_fila_id", "MasterDate"], errors="ignore")
    logger.info("  Actividades integradas: %d filas con asistencias acumuladas",
                (df_final["num_asistencias_acum"] > 0).sum())
    return df_final


# ===========================================================================
# VALIDACIÓN DE CAMPOS: notebook vs Oracle
# ===========================================================================

def validar_campos(df_opp: pd.DataFrame, df_acc: pd.DataFrame,
                   df_ecbs: pd.DataFrame, df_stage: pd.DataFrame) -> pd.DataFrame:
    """
    Compara los campos esperados por el notebook con las columnas disponibles
    en Oracle. Devuelve un DataFrame con el estado de cada campo.
    """
    # Columnas disponibles tras renombrado
    cols_disp = (
        set(df_opp.columns) |
        set(df_acc.columns) |
        set(df_ecbs.columns) |
        set(df_stage.columns)
    )

    rows = []
    for nb_name, (tabla, ora_col, nota) in NOTEBOOK_FIELDS.items():
        if tabla in ("COMPUTED", "—"):
            estado = "🔧 derivado" if tabla == "COMPUTED" else "❌ no existe en Oracle"
        elif nb_name in cols_disp or (ora_col and ora_col in cols_disp):
            estado = "✅ disponible"
        else:
            estado = "⚠️ columna no encontrada tras renombrado"

        rows.append({
            "Nombre notebook":  nb_name,
            "Tabla Oracle":     tabla,
            "Columna Oracle":   ora_col or "—",
            "Nota":             nota,
            "Estado":           estado,
        })

    df = pd.DataFrame(rows)
    ok     = (df["Estado"] == "✅ disponible").sum()
    der    = df["Estado"].str.startswith("🔧").sum()
    ko     = df["Estado"].str.startswith("❌").sum()
    warn   = df["Estado"].str.startswith("⚠️").sum()

    logger.info("  Campo notebook vs Oracle:")
    logger.info("    ✅ disponible: %d", ok)
    logger.info("    🔧 derivado:   %d", der)
    logger.info("    ❌ ausente:    %d", ko)
    logger.info("    ⚠️ aviso:      %d", warn)
    return df


# ===========================================================================
# GENERACIÓN DE DOCUMENTACIÓN
# ===========================================================================

def _write_docs(
    df_validacion:  pd.DataFrame,
    metricas:       dict,
    start_ts:       datetime,
    end_ts:         datetime,
    recreate:       bool,
) -> Path:
    """Genera docs/FASE2_LIMPIEZA.md con comparativa y métricas de limpieza."""
    dur = (end_ts - start_ts).total_seconds()

    lines: list[str] = [
        "# Fase 2 – Limpieza y construcción del dataset de modelado",
        "",
        "## ¿Qué hace esta fase?",
        "",
        "Lee las 10 tablas de Oracle cargadas en Fase 1, aplica la lógica de limpieza",
        "del notebook `01_Limpieza copy.ipynb` y genera la tabla `DATASET_LIMPIO`",
        "lista para modelado con PyCaret.",
        "",
        "### Pipeline de limpieza (15 pasos)",
        "",
        "| Paso | Descripción |",
        "|------|-------------|",
        "| 1  | Carga de tablas desde Oracle (OPPORTUNITY, ACCOUNT, ECBS, STAGE_HISTORY, ACTIVITY_HISTORY) |",
        "| 2  | Renombrado de columnas Oracle → convención notebook |",
        "| 3  | Validación de campos: notebook vs Oracle |",
        "| 4  | Eliminación de columnas con >90% de NA |",
        "| 5  | Creación del target (Matrícula OOGG formalizada) y flag desmatriculado |",
        "| 6  | JOIN Opportunity × Account (LEFT, clave ACCOUNTID = ID) |",
        "| 7  | Variables derivadas: YEARPERSONBIRTHDATE, TITULACION, CENTROENSENANZA, RECORDTYPENAME, PL_TIPOSOLICITUD |",
        "| 8  | Normalización del plazo de admisión: Diciembre / Marzo / Rolling / Otros |",
        "| 9  | JOIN con ECBS + cálculo de PORCENTAJE_PAGADO_FINAL |",
        "| 10 | Cálculo de tiempos en cada etapa del funnel (tiempo_etapa_dias, tiempo_entre_etapas_dias) |",
        "| 11 | Control de leakage por hitos temporales (pruebas calificadas, pago mínimo) |",
        "| 12 | Integración de actividades acumuladas (num_asistencias_acum, num_solicitudes_acum) |",
        "| 13 | Detección y corrección de leakage de pago en etapas tempranas |",
        "| 14 | Eliminación de registros sin target válido |",
        "| 15 | Selección de 56 columnas finales y guardado en Oracle |",
        "",
        "## Ejecución",
        "",
        f"- **Fecha/hora inicio:** {start_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Fecha/hora fin:**    {end_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Duración total:**    {dur:.1f} s",
        f"- **Modo:**              {'RECREAR tabla (--recreate)' if recreate else 'Insert acumulativo'}",
        "",
        "## Métricas del dataset",
        "",
        "| Métrica | Valor |",
        "|---------|-------|",
        f"| Filas tras join (antes limpieza) | {metricas.get('filas_raw', '?'):,} |",
        f"| Filas con target nulo eliminadas | {metricas.get('filas_target_nulo', '?'):,} |",
        f"| Filas finales en DATASET_LIMPIO  | {metricas.get('filas_final', '?'):,} |",
        f"| Columnas seleccionadas           | {metricas.get('cols_final', '?')} |",
        f"| Target = 1 (matriculados)        | {metricas.get('target_1', '?'):,} |",
        f"| Target = 0 (no matriculados)     | {metricas.get('target_0', '?'):,} |",
        f"| Tasa conversión                  | {metricas.get('tasa_conversion', '?')} % |",
        "",
        "## Comparativa: campos notebook vs Oracle",
        "",
        "Esta tabla muestra si cada campo esperado por el notebook existe en Oracle,",
        "si necesita renombrado, si es derivado o si está ausente.",
        "",
        "| Campo notebook | Tabla Oracle | Columna Oracle | Estado | Nota |",
        "|---|---|---|---|---|",
    ]

    for _, row in df_validacion.iterrows():
        lines.append(
            f"| `{row['Nombre notebook']}` | {row['Tabla Oracle']} "
            f"| `{row['Columna Oracle']}` | {row['Estado']} | {row['Nota']} |"
        )

    # Diferencias destacadas
    df_warn = df_validacion[~df_validacion["Estado"].str.startswith("✅") &
                             ~df_validacion["Estado"].str.startswith("🔧")]
    if not df_warn.empty:
        lines += [
            "",
            "## Diferencias y campos no disponibles",
            "",
        ]
        for _, row in df_warn.iterrows():
            lines.append(f"- **`{row['Nombre notebook']}`** — {row['Estado']}: {row['Nota']}")

    lines += [
        "",
        "## Notas técnicas",
        "",
        "- **Leakage académico**: las columnas de notas y resolución se ponen a NaN",
        "  si la fila del historial es anterior a la fecha en que se calificaron las pruebas.",
        "- **Leakage económico**: las columnas de pago se ponen a NaN si la fila es",
        "  anterior al primer pago mínimo.",
        "- **NAMEX**: el campo nombre no fue seleccionado en la query de Salesforce.",
        "  Se puede añadir en fases futuras si se necesita.",
        "- **CH_ANTIGUO_ALUMNO__PC**: en Oracle la columna se llama `CH_ANTIGUOALUMNO__PC`",
        "  (sin guión bajo entre ANTIGUO y ALUMNO). Se renombra al cargar.",
        "- **Truncados**: CH_HERMANOS_ESTUDIANDO_UNAV__P y NU_MEDIA_EXPEDIENTE_UNIVERSITA",
        "  aparecen truncados en el notebook (nombres de hasta 30 chars del Excel).",
        "  En Oracle tienen su nombre completo.",
        f"- **Log completo**: `logs/cleaner.log`",
        "",
        "---",
        f"*Generado automáticamente el {end_ts.strftime('%Y-%m-%d %H:%M')} UTC*",
    ]

    doc_path = DOCS_DIR / "FASE2_LIMPIEZA.md"
    doc_path.write_text("\n".join(lines), encoding="utf-8")
    return doc_path


# ===========================================================================
# PIPELINE PRINCIPAL
# ===========================================================================

# Columnas finales del dataset de modelado (últimas columnas_seleccionadas del notebook)
COLUMNAS_FINALES = [
    "ACCOUNTID", "ID", "ID18__PC", "target", "desmatriculado",
    "PL_CURSO_ACADEMICO", "CH_NACIONAL",
    "NU_NOTA_MEDIA_ADMISION", "NU_NOTA_MEDIA_1_BACH__PC",
    "CH_PRUEBAS_CALIFICADAS", "NU_RESULTADO_ADMISION_PUNTOS",
    "PL_RESOLUCION_DEFINITIVA", "TITULACION", "CENTROENSENANZA",
    "MINIMUMPAYMENTPAYED", "PAID_AMOUNT", "CH_PAGO_SUPERIOR",
    "CH_MATRICULA_SUJETA_BECA", "CH_AYUDA_FINANCIACION", "CU_IMPORTE_TOTAL",
    "CH_VISITACAMPUS__PC", "CH_ENTREVISTA_PERSONAL__PC",
    "ACC_DTT_FECHAULTIMAACTIVIDAD", "NU_PREFERENCIA",
    "PL_Etapa__c", "PL_Subetapa__c",
    "CH_HIJO_EMPLEADO__PC", "CH_HIJO_PROFESOR_ASOCIADO__PC",
    "CH_HERMANOS_ESTUDIANDO_UNAV__P", "CH_HIJO_MEDICO__PC",
    "YEARPERSONBIRTHDATE", "CH_FAMILIA_NUMEROSA__PC",
    "PL_SITUACION_SOCIO_ECONOMICA", "LEADSOURCE",
    "PL_ORIGEN_DE_SOLICITUD", "PL_PLAZO_ADMISION",
    "RECORDTYPEID", "RECORDTYPENAME", "PLAZO_ADMISION_LIMPIO",
    "FO_rentaFam_ges__c", "CU_precioOrdinario_def__c",
    "CU_precioAplicado_def__c", "PORCENTAJE_PAGADO_FINAL",
    "tiempo_etapa_dias", "tiempo_entre_etapas_dias",
    "num_asistencias_acum", "num_solicitudes_acum",
    "CH_ALUMNO__PC", "CH_ESTUDIANTE__PC", "CH_ANTIGUO_ALUMNO__PC",
    "CH_ALUMNI__PC", "CH_ANTIGUOALUMNO_INTERCAMBIO",
    "CH_HIJO_ANTIGUO_ALUMNO__PC",
    "CreatedDate", "NU_MEDIA_EXPEDIENTE_UNIVERSITA",
    "FUENTE_DATOS",  # 'SF_2026_27' o 'HISTORICO' — permite filtrar para entrenar/predecir
]


def _tabla_existe(ora: OracleConnector, tabla: str) -> bool:
    """Devuelve True si la tabla existe en el schema Oracle."""
    try:
        cur = ora.conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM all_tables WHERE owner=:s AND table_name=:t",
            s=ora.schema.upper(), t=tabla.upper(),
        )
        existe = cur.fetchone()[0] > 0
        cur.close()
        return existe
    except Exception:
        return False


def _cargar_historico(ora: OracleConnector) -> pd.DataFrame | None:
    """
    Carga y procesa las tablas históricas (_HIST) cargadas desde el Excel.
    Aplica el mismo pipeline de limpieza que los datos actuales de SF, con
    las adaptaciones necesarias para el formato del Excel:
      - Join Account usa ACCOUNTID ↔ ID18 (18 chars, no ID de 15 chars)
      - YEARPERSONBIRTHDATE ya calculado en ACCOUNT_HIST
      - TITULACION viene de la columna TITULACION_DEF (renombrada)
      - CENTROENSENANZA viene directamente de ACCOUNT_HIST

    Returns:
        DataFrame expandido (1 fila por etapa × oportunidad) o None si las
        tablas HIST no existen en Oracle (excel_loader aún no ejecutado).
    """
    tablas_requeridas = [
        "OPPORTUNITY_HIST", "ACCOUNT_HIST", "ECBS_HIST",
        "STAGE_HISTORY_HIST", "ACTIVITY_HIST"
    ]
    faltan = [t for t in tablas_requeridas if not _tabla_existe(ora, t)]
    if faltan:
        logger.warning(
            "Tablas HIST no encontradas: %s. "
            "Ejecuta primero: python src/excel_loader.py --recreate",
            faltan,
        )
        return None

    logger.info("─" * 60)
    logger.info("  [HISTÓRICO] Cargando tablas HIST desde Oracle...")

    def load_hist(tabla: str) -> pd.DataFrame:
        rows = ora.read_table(tabla)
        df = pd.DataFrame(rows)
        logger.info("    %s: %d filas × %d cols", tabla, *df.shape)
        return df

    df_opp_h   = load_hist("OPPORTUNITY_HIST")
    df_acc_h   = load_hist("ACCOUNT_HIST")
    df_ecbs_h  = load_hist("ECBS_HIST")
    df_stage_h = load_hist("STAGE_HISTORY_HIST")
    df_act_h   = load_hist("ACTIVITY_HIST")

    # Renames
    df_opp_h   = df_opp_h.rename(columns={k: v for k, v in OPP_HIST_RENAME.items()   if k in df_opp_h.columns})
    df_ecbs_h  = df_ecbs_h.rename(columns={k: v for k, v in ECBS_HIST_RENAME.items()  if k in df_ecbs_h.columns})
    df_stage_h = df_stage_h.rename(columns={k: v for k, v in STAGE_HIST_RENAME.items() if k in df_stage_h.columns})
    df_act_h   = df_act_h.rename(columns={k: v for k, v in ACT_HIST_RENAME.items()   if k in df_act_h.columns})

    # Eliminar columnas >90% NA
    df_opp_h  = eliminar_columnas_na(df_opp_h)
    df_acc_h  = eliminar_columnas_na(df_acc_h)
    df_ecbs_h = eliminar_columnas_na(df_ecbs_h)

    logger.info(
        "    Tras NA >90%%: opp=%d cols, acc=%d cols, ecbs=%d cols",
        len(df_opp_h.columns), len(df_acc_h.columns), len(df_ecbs_h.columns),
    )

    # Target (usa historial con columnas renombradas: PL_Etapa__c, PL_Subetapa__c, LK_Oportunidad__c)
    df_opp_h = crear_target(df_opp_h, df_stage_h)
    n_target1 = int(df_opp_h["target"].sum())
    logger.info("    [HISTÓRICO] Target=1: %d | Target=0: %d",
                n_target1, len(df_opp_h) - n_target1)

    # JOIN Oportunidad × Cuenta (clave ID18 en ACCOUNT_HIST)
    acc_join_key = "ID18" if "ID18" in df_acc_h.columns else "ID"
    df_unido_h = pd.merge(
        df_opp_h, df_acc_h,
        left_on="ACCOUNTID", right_on=acc_join_key,
        how="left", suffixes=("", "_cuenta")
    )
    dup_cols = [c for c in df_unido_h.columns if c.endswith("_cuenta")]
    df_unido_h = df_unido_h.drop(columns=dup_cols)
    logger.info("    [HISTÓRICO] JOIN opp×cuenta: %d filas × %d cols",
                *df_unido_h.shape)

    # YEARPERSONBIRTHDATE: ya presente en ACCOUNT_HIST; derivar si no está
    if "YEARPERSONBIRTHDATE" not in df_unido_h.columns:
        df_unido_h["YEARPERSONBIRTHDATE"] = pd.to_datetime(
            df_unido_h.get("PERSONBIRTHDATE"), errors="coerce"
        ).dt.year

    # PL_TIPOSOLICITUD
    if "RECORDTYPENAME" in df_unido_h.columns:
        df_unido_h["PL_TIPOSOLICITUD"] = np.select(
            [
                df_unido_h["RECORDTYPENAME"].str.contains("rado", case=False, na=False),
                df_unido_h["RECORDTYPENAME"].str.contains("ster", case=False, na=False),
            ],
            ["Grado", "Máster"],
            default="Otro",
        )

    # PLAZO_ADMISION_LIMPIO (misma lógica que datos actuales)
    def _normalizar_plazo_h(row) -> str:
        tipo = str(row.get("PL_TIPOSOLICITUD", "")).lower()
        if "ster" in tipo:
            return "Rolling"
        plazo = row.get("PL_PLAZO_ADMISION")
        if pd.isna(plazo):
            return "Rolling"
        plazo = str(plazo).strip().lower()
        if "dic" in plazo:
            return "Diciembre"
        if "mar" in plazo:
            return "Marzo"
        return "Otros"

    df_unido_h["PLAZO_ADMISION_LIMPIO"] = df_unido_h.apply(_normalizar_plazo_h, axis=1)

    # JOIN con ECBS
    ecbs_vars = ["LK_oportunidad__c", "FO_rentaFam_ges__c",
                 "CU_precioOrdinario_def__c", "CU_precioAplicado_def__c"]
    ecbs_vars = [v for v in ecbs_vars if v in df_ecbs_h.columns]
    ecbs_join_key = "LK_oportunidad__c" if "LK_oportunidad__c" in df_ecbs_h.columns else "ID"
    df_unido_h = pd.merge(
        df_unido_h,
        df_ecbs_h[ecbs_vars],
        left_on="ID", right_on=ecbs_join_key,
        how="left"
    )

    if "CU_precioAplicado_def__c" in df_unido_h.columns and \
       "CU_precioOrdinario_def__c" in df_unido_h.columns:
        df_unido_h["PORCENTAJE_PAGADO_FINAL"] = (
            pd.to_numeric(df_unido_h["CU_precioAplicado_def__c"], errors="coerce") /
            pd.to_numeric(df_unido_h["CU_precioOrdinario_def__c"], errors="coerce")
        ) * 100
        mask_inv = (
            (pd.to_numeric(df_unido_h["CU_precioOrdinario_def__c"], errors="coerce") <= 0) |
            (df_unido_h["PORCENTAJE_PAGADO_FINAL"] < 0) |
            (df_unido_h["PORCENTAJE_PAGADO_FINAL"] > 100)
        )
        df_unido_h.loc[mask_inv, "PORCENTAJE_PAGADO_FINAL"] = np.nan

    # Calcular tiempos en etapa
    df_stage_h_t = calcular_tiempos_etapas(df_stage_h)
    logger.info("    [HISTÓRICO] tiempo_etapa_dias: min=%d, max=%d, media=%.1f",
                df_stage_h_t["tiempo_etapa_dias"].min(),
                df_stage_h_t["tiempo_etapa_dias"].max(),
                df_stage_h_t["tiempo_etapa_dias"].mean())

    # Expansión por hitos (leakage temporal) + expand 1 fila por etapa
    cols_etapa = [c for c in ["PL_SUBETAPA", "STAGENAME"] if c in df_unido_h.columns]
    df_unido_h_sin_etapa = df_unido_h.drop(columns=cols_etapa)
    df_hist_expandido = limpiar_historial_por_hitos(df_stage_h_t, df_unido_h_sin_etapa)
    logger.info("    [HISTÓRICO] Dataset expandido: %d filas × %d cols",
                *df_hist_expandido.shape)

    # Integrar actividades acumuladas
    df_hist_expandido = integrar_actividades_progresivo(df_hist_expandido, df_act_h)

    # Etiquetar la fuente de datos
    df_hist_expandido["FUENTE_DATOS"] = "HISTORICO"

    return df_hist_expandido


def run(recreate: bool = False, include_historical: bool = False) -> None:
    start_ts = datetime.now(timezone.utc)
    metricas: dict = {}
    TOTAL_STEPS = 15

    logger.info("=" * 70)
    logger.info("FASE 2 – LIMPIEZA Y CONSTRUCCIÓN DEL DATASET  [INICIO]")
    logger.info("=" * 70)

    with OracleConnector() as ora:

        # ── 1. CARGA DESDE ORACLE ──────────────────────────────────────────
        _step(1, TOTAL_STEPS, "Carga de tablas desde Oracle")

        def load(table: str) -> pd.DataFrame:
            rows = ora.read_table(table)
            df = pd.DataFrame(rows)
            logger.info("  %s: %d filas × %d cols", table, *df.shape)
            return df

        df_opp   = load("OPPORTUNITY")
        df_acc   = load("ACCOUNT")
        df_ecbs  = load("ECBS")
        df_stage = load("STAGE_HISTORY")
        df_act   = load("ACTIVITY_HISTORY")

        # ── 2. RENOMBRADO DE COLUMNAS ──────────────────────────────────────
        _step(2, TOTAL_STEPS, "Renombrado de columnas Oracle → convención notebook")

        df_opp   = df_opp.rename(columns={k: v for k, v in OPP_RENAME.items()   if k in df_opp.columns})
        df_acc   = df_acc.rename(columns={k: v for k, v in ACC_RENAME.items()   if k in df_acc.columns})
        df_ecbs  = df_ecbs.rename(columns={k: v for k, v in ECBS_RENAME.items() if k in df_ecbs.columns})
        df_stage = df_stage.rename(columns={k: v for k, v in STAGE_RENAME.items() if k in df_stage.columns})
        df_act   = df_act.rename(columns={k: v for k, v in ACT_RENAME.items()   if k in df_act.columns})

        logger.info("  OPPORTUNITY: %d cols", len(df_opp.columns))
        logger.info("  ACCOUNT:     %d cols", len(df_acc.columns))
        logger.info("  ECBS:        %d cols", len(df_ecbs.columns))
        logger.info("  STAGE_HISTORY: %d cols", len(df_stage.columns))
        logger.info("  ACTIVITY_HISTORY: %d cols", len(df_act.columns))

        # ── 3. VALIDACIÓN DE CAMPOS ────────────────────────────────────────
        _step(3, TOTAL_STEPS, "Validación de campos notebook vs Oracle")
        df_validacion = validar_campos(df_opp, df_acc, df_ecbs, df_stage)

        # ── 4. ELIMINAR COLUMNAS CON EXCESO DE NA ─────────────────────────
        _step(4, TOTAL_STEPS, "Eliminación de columnas con >90% de NA")
        before = {n: len(d.columns) for n, d in [
            ("OPPORTUNITY", df_opp), ("ACCOUNT", df_acc),
            ("ECBS", df_ecbs), ("STAGE_HISTORY", df_stage)
        ]}
        df_opp   = eliminar_columnas_na(df_opp)
        df_acc   = eliminar_columnas_na(df_acc)
        df_ecbs  = eliminar_columnas_na(df_ecbs)
        df_stage = eliminar_columnas_na(df_stage)
        for n, df in [("OPPORTUNITY", df_opp), ("ACCOUNT", df_acc),
                      ("ECBS", df_ecbs), ("STAGE_HISTORY", df_stage)]:
            logger.info("  %s: %d → %d columnas", n, before[n], len(df.columns))

        # ── 5. CREACIÓN DEL TARGET ─────────────────────────────────────────
        _step(5, TOTAL_STEPS, "Creación del target y flag desmatriculado")
        df_opp = crear_target(df_opp, df_stage)

        # ── 6. JOIN OPPORTUNITY × ACCOUNT ─────────────────────────────────
        _step(6, TOTAL_STEPS, "JOIN Opportunity × Account")
        df_unido = pd.merge(
            df_opp, df_acc,
            left_on="ACCOUNTID", right_on="ID",
            how="left",
            suffixes=("", "_cuenta")
        )
        # Eliminar columnas duplicadas (_cuenta)
        dup_cols = [c for c in df_unido.columns if c.endswith("_cuenta")]
        df_unido = df_unido.drop(columns=dup_cols)
        logger.info("  Resultado JOIN: %d filas × %d columnas", *df_unido.shape)

        # ── 7. VARIABLES DERIVADAS ─────────────────────────────────────────
        _step(7, TOTAL_STEPS, "Construcción de variables derivadas")

        # Año de nacimiento
        df_unido["YEARPERSONBIRTHDATE"] = pd.to_datetime(
            df_unido.get("PERSONBIRTHDATE"), errors="coerce"
        ).dt.year
        logger.info("  YEARPERSONBIRTHDATE: %d no nulos",
                    df_unido["YEARPERSONBIRTHDATE"].notna().sum())

        # Tipo solicitud (Grado / Máster / Otro)
        if "RECORDTYPENAME" in df_unido.columns:
            df_unido["PL_TIPOSOLICITUD"] = np.select(
                [
                    df_unido["RECORDTYPENAME"].str.contains("rado", case=False, na=False),
                    df_unido["RECORDTYPENAME"].str.contains("ster", case=False, na=False),
                ],
                ["Grado", "Máster"],
                default="Otro"
            )
            logger.info("  PL_TIPOSOLICITUD: %s",
                        df_unido["PL_TIPOSOLICITUD"].value_counts().to_dict())

        # Titulación definitiva (ya renombrada en OPP_RENAME)
        if "TITULACION" not in df_unido.columns:
            # Intentar con LK_TITULACION__R.NAME si la definitiva es nula
            alt = "LK_TITULACION__R.NAME"
            if alt in df_unido.columns:
                df_unido["TITULACION"] = df_unido[alt]
                logger.info("  TITULACION: usando columna alternativa %s", alt)
            else:
                df_unido["TITULACION"] = None
                logger.warning("  TITULACION no disponible")

        # ── 8. NORMALIZACIÓN PLAZO ADMISIÓN ───────────────────────────────
        _step(8, TOTAL_STEPS, "Normalización del plazo de admisión")

        def normalizar_plazo(row) -> str:
            tipo = str(row.get("PL_TIPOSOLICITUD", "")).lower()
            if "ster" in tipo:
                return "Rolling"
            plazo = row.get("PL_PLAZO_ADMISION")
            if pd.isna(plazo):
                return "Rolling"
            plazo = str(plazo).strip().lower()
            if "dic" in plazo:
                return "Diciembre"
            if "mar" in plazo:
                return "Marzo"
            return "Otros"

        df_unido["PLAZO_ADMISION_LIMPIO"] = df_unido.apply(normalizar_plazo, axis=1)
        logger.info("  Distribución PLAZO_ADMISION_LIMPIO: %s",
                    df_unido["PLAZO_ADMISION_LIMPIO"].value_counts().to_dict())

        # ── 9. JOIN CON ECBS + PORCENTAJE PAGADO ──────────────────────────
        _step(9, TOTAL_STEPS, "JOIN con ECBS + PORCENTAJE_PAGADO_FINAL")

        ecbs_vars = ["LK_oportunidad__c", "FO_rentaFam_ges__c",
                     "CU_precioOrdinario_def__c", "CU_precioAplicado_def__c"]
        ecbs_vars = [v for v in ecbs_vars if v in df_ecbs.columns]

        df_unido = pd.merge(
            df_unido,
            df_ecbs[ecbs_vars],
            left_on="ID", right_on="LK_oportunidad__c",
            how="left"
        )

        if "CU_precioAplicado_def__c" in df_unido and "CU_precioOrdinario_def__c" in df_unido:
            df_unido["PORCENTAJE_PAGADO_FINAL"] = (
                pd.to_numeric(df_unido["CU_precioAplicado_def__c"],  errors="coerce") /
                pd.to_numeric(df_unido["CU_precioOrdinario_def__c"], errors="coerce")
            ) * 100
            # Invalidar valores fuera de rango
            mask_invalid = (
                (df_unido["CU_precioOrdinario_def__c"] <= 0) |
                (df_unido["PORCENTAJE_PAGADO_FINAL"] < 0)   |
                (df_unido["PORCENTAJE_PAGADO_FINAL"] > 100)
            )
            df_unido.loc[mask_invalid, "PORCENTAJE_PAGADO_FINAL"] = np.nan
            logger.info("  PORCENTAJE_PAGADO_FINAL: %d valores válidos, %d NaN",
                        df_unido["PORCENTAJE_PAGADO_FINAL"].notna().sum(),
                        df_unido["PORCENTAJE_PAGADO_FINAL"].isna().sum())

        # ── 10. TIEMPOS EN ETAPAS ──────────────────────────────────────────
        _step(10, TOTAL_STEPS, "Cálculo de tiempos en cada etapa (calcular_tiempos_etapas)")
        df_stage_t = calcular_tiempos_etapas(df_stage)
        logger.info("  tiempo_etapa_dias: min=%d, max=%d, media=%.1f",
                    df_stage_t["tiempo_etapa_dias"].min(),
                    df_stage_t["tiempo_etapa_dias"].max(),
                    df_stage_t["tiempo_etapa_dias"].mean())

        # ── 11. CONTROL DE LEAKAGE TEMPORAL ───────────────────────────────
        _step(11, TOTAL_STEPS, "Control de leakage temporal por hitos (limpiar_historial_por_hitos)")
        # Eliminar columnas de etapa estáticas del df_unido antes del merge
        cols_etapa = [c for c in ["PL_SUBETAPA", "STAGENAME"] if c in df_unido.columns]
        df_unido_sin_etapa = df_unido.drop(columns=cols_etapa)

        df_definitivo = limpiar_historial_por_hitos(df_stage_t, df_unido_sin_etapa)
        df_definitivo["FUENTE_DATOS"] = "SF_2026_27"
        logger.info("  Dataset expandido (1 fila por etapa × oportunidad): %d filas", len(df_definitivo))

        # ── 11b. COMBINACIÓN CON DATOS HISTÓRICOS (opcional) ───────────────
        if include_historical:
            logger.info("─" * 60)
            logger.info("PASO 11b │ Carga y combinación con datos históricos del Excel")
            logger.info("─" * 60)
            df_hist = _cargar_historico(ora)
            if df_hist is not None:
                n_actual = len(df_definitivo)
                n_hist   = len(df_hist)
                df_definitivo = pd.concat(
                    [df_definitivo, df_hist], ignore_index=True, sort=False
                )
                metricas["filas_actuales"]  = n_actual
                metricas["filas_historico"] = n_hist
                logger.info(
                    "  Datos actuales: %d filas | Histórico: %d filas | Total: %d filas",
                    n_actual, n_hist, len(df_definitivo),
                )
            else:
                logger.warning("  Sin datos históricos disponibles. Continuando solo con datos actuales.")

        metricas["filas_raw"] = len(df_definitivo)

        # ── 12. ACTIVIDADES ACUMULADAS ─────────────────────────────────────
        _step(12, TOTAL_STEPS, "Integración de actividades acumuladas")
        # Las actividades ya se integran dentro de _cargar_historico para el histórico;
        # aquí solo se procesan los datos actuales (SF_2026_27).
        if not include_historical:
            df_definitivo = integrar_actividades_progresivo(df_definitivo, df_act)
        else:
            # Los datos actuales aún no tienen actividades; los históricos ya las tienen.
            df_actual = df_definitivo[df_definitivo["FUENTE_DATOS"] == "SF_2026_27"].copy()
            df_hist_ya = df_definitivo[df_definitivo["FUENTE_DATOS"] == "HISTORICO"].copy()
            df_actual = integrar_actividades_progresivo(df_actual, df_act)
            df_definitivo = pd.concat([df_actual, df_hist_ya], ignore_index=True, sort=False)

        # ── 13. LEAKAGE DE PAGO EN ETAPAS TEMPRANAS ───────────────────────
        _step(13, TOTAL_STEPS, "Corrección de leakage de pago en etapas tempranas")
        etapas_sin_pago = ["Solicitud", "Pruebas", "Admisión académica"]
        vars_pago = [v for v in [
            "PAID_AMOUNT", "MINIMUMPAYMENTPAYED",
            "CU_precioAplicado_def__c", "PORCENTAJE_PAGADO_FINAL"
        ] if v in df_definitivo.columns]

        if "PL_Etapa__c" in df_definitivo.columns and vars_pago:
            mask_leakage = (
                df_definitivo["PL_Etapa__c"].isin(etapas_sin_pago) &
                df_definitivo[vars_pago].notna().any(axis=1)
            )
            df_definitivo.loc[mask_leakage, vars_pago] = np.nan
            logger.info("  Leakage de pago corregido: %d filas", mask_leakage.sum())

        # ── 14. ELIMINAR REGISTROS SIN TARGET ─────────────────────────────
        _step(14, TOTAL_STEPS, "Eliminación de registros con target nulo")
        antes = len(df_definitivo)
        df_definitivo = df_definitivo[df_definitivo["target"].notna()].copy()
        nulos_eliminados = antes - len(df_definitivo)
        metricas["filas_target_nulo"] = nulos_eliminados
        logger.info("  Eliminados %d registros con target nulo", nulos_eliminados)
        logger.info("  Target=1: %d | Target=0: %d",
                    df_definitivo["target"].sum(),
                    (df_definitivo["target"] == 0).sum())

        # ── 15. SELECCIÓN FINAL Y GUARDADO EN ORACLE ──────────────────────
        _step(15, TOTAL_STEPS, "Selección de columnas finales y guardado en Oracle")

        # Solo columnas que existen en el DataFrame
        cols_disponibles = [c for c in COLUMNAS_FINALES if c in df_definitivo.columns]
        cols_faltantes   = [c for c in COLUMNAS_FINALES if c not in df_definitivo.columns]

        if cols_faltantes:
            logger.warning("  Columnas finales NO encontradas en dataset: %s", cols_faltantes)

        df_final = df_definitivo[cols_disponibles].copy()
        metricas["filas_final"]       = len(df_final)
        metricas["cols_final"]        = len(cols_disponibles)
        metricas["target_1"]          = int(df_final["target"].sum())
        metricas["target_0"]          = int((df_final["target"] == 0).sum())
        metricas["tasa_conversion"]   = round(
            metricas["target_1"] / len(df_final) * 100, 2) if len(df_final) else 0

        logger.info("  Dataset final: %d filas × %d columnas", *df_final.shape)
        logger.info("  Target=1: %d (%.1f%%)", metricas["target_1"], metricas["tasa_conversion"])

        # Convertir todos los valores a tipos Python nativos (evitar numpy types)
        records = df_final.astype(object).where(df_final.notna(), None).to_dict("records")

        # Guardar en Oracle
        if recreate:
            ora.drop_table_if_exists("DATASET_LIMPIO")
        else:
            ora.truncate_table("DATASET_LIMPIO")

        metrics_ora = ora.insert_records(records, "DATASET_LIMPIO")
        logger.info("  DATASET_LIMPIO → %d filas insertadas en Oracle", metrics_ora["inserted"])

    # ── RESUMEN FINAL ──────────────────────────────────────────────────────
    end_ts = datetime.now(timezone.utc)
    dur = (end_ts - start_ts).total_seconds()

    logger.info("=" * 70)
    logger.info("FASE 2 – LIMPIEZA Y CONSTRUCCIÓN DEL DATASET  [FIN]")
    logger.info("=" * 70)
    logger.info("")
    logger.info("  Filas expandidas (opp × etapas):  %10d", metricas.get("filas_raw", 0))
    logger.info("  Filas con target nulo eliminadas: %10d", metricas.get("filas_target_nulo", 0))
    logger.info("  Filas finales en DATASET_LIMPIO:  %10d", metricas.get("filas_final", 0))
    logger.info("  Columnas seleccionadas:           %10d", metricas.get("cols_final", 0))
    logger.info("  Target=1 (matriculados):          %10d  (%.1f%%)",
                metricas.get("target_1", 0), metricas.get("tasa_conversion", 0))
    logger.info("  Duración: %.1f s", dur)

    # ── DOCUMENTACIÓN ──────────────────────────────────────────────────────
    doc_path = _write_docs(df_validacion, metricas, start_ts, end_ts, recreate)
    logger.info("Documentación generada: %s", doc_path)


def main(args=None):
    """Entry point programático para la pipeline orquestadora."""
    parser = argparse.ArgumentParser(description="Fase 2: Limpieza y dataset de modelado")
    parser.add_argument(
        "--recreate", action="store_true",
        help="Elimina y recrea DATASET_LIMPIO antes de insertar"
    )
    parser.add_argument(
        "--include-historical", action="store_true",
        help=(
            "Combina los datos actuales de SF (2026/27) con el histórico del Excel "
            "(tablas *_HIST cargadas por excel_loader.py). Necesario para entrenar el modelo."
        )
    )
    parsed = parser.parse_args(args)
    run(recreate=parsed.recreate, include_historical=parsed.include_historical)


if __name__ == "__main__":
    main()
