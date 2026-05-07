"""
Fase 4 - Tests unitarios del módulo preprocessor.py

Todos los tests usan DataFrames sintéticos — NO requieren conexión Oracle
ni PyCaret instalado.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Asegurar que src/ está en el path para importar preprocessor
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessor import (
    split_grado_master,
    calcular_orden_automatico,
    imputar,
    crear_vinculacion_previa,
    get_safe_cols,
    preprocess,
    VARS_CERO_LOGICO,
    COLS_VINCULACION,
    COLS_ID,
    VARS_EXCLUIR,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_df_base(n_grado=5, n_master=3) -> pd.DataFrame:
    """DataFrame mínimo que simula DATASET_LIMPIO."""
    total = n_grado + n_master
    titulaciones = ["Grado en Medicina"] * n_grado + ["MASTER en Salud"] * n_master

    # Listas cíclicas truncadas exactamente a 'total'
    etapas = (["Lead", "Admisión", "Matrícula"] * ((total // 3) + 2))[:total]
    subetapas = (["Inicial", "Admitido", "OOGG"] * ((total // 3) + 2))[:total]
    notas = ([8.5, np.nan, 7.0, 9.0, np.nan, 8.0, 7.5, 8.8] * 4)[:total]
    notas_bach = ([7.0, 8.0, np.nan, 6.5, 9.0, 7.5, 8.0, 7.0] * 4)[:total]
    importes = ([5000, np.nan, 4500, 5200, 4800, 6000, 5500, 4900] * 4)[:total]
    paid = ([0.5, 1.0, np.nan, 0.75, 0.0, 0.8, 0.6, 0.9] * 4)[:total]
    renta = ([30000, np.nan, 25000, np.nan, 40000, 35000, np.nan, 28000] * 4)[:total]
    beca = ([1, np.nan, 0, 1, np.nan, 0, 1, 0] * 4)[:total]
    alumno = ([True, False, False, True, False, False, True, False] * 4)[:total]
    estudiante = ([False, True, False, False, True, False, False, True] * 4)[:total]
    antiguo = ([False, False, True, False, False, True, False, False] * 4)[:total]
    target_vals = ([1, 0] * ((total // 2) + 2))[:total]
    preferencia = ([1, 2, 1, 3, 1, 2, 1, 2] * 4)[:total]

    df = pd.DataFrame({
        "ID": [f"OPP{i:03d}" for i in range(total)],
        "ACCOUNTID": [f"ACC{i:03d}" for i in range(total)],
        "ID18__PC": [f"ID18_{i}" for i in range(total)],
        "BIRTHDATE": ["1990-01-01"] * total,
        "CreatedDate": pd.date_range("2023-01-01", periods=total, freq="D"),
        "TITULACION": titulaciones,
        "PL_Etapa__c": etapas,
        "PL_Subetapa__c": subetapas,
        "NU_NOTA_MEDIA_ADMISION": notas,
        "NU_NOTA_MEDIA_1_BACH__PC": notas_bach,
        "CU_IMPORTE_TOTAL": importes,
        "PAID_PERCENT": paid,
        "FO_rentaFam_ges__c": renta,
        "CH_MATRICULA_SUJETA_BECA": beca,
        "CH_ALUMNO__PC": alumno,
        "CH_ESTUDIANTE__PC": estudiante,
        "CH_ANTIGUO_ALUMNO__PC": antiguo,
        "CH_ALUMNI__PC": [False] * total,
        "CH_ANTIGUOALUMNO_INTERCAMBIO": [False] * total,
        "CH_HIJO_ANTIGUO_ALUMNO__PC": [False] * total,
        "target": target_vals,
        "desmatriculado": [0] * total,
        "PL_CURSO_ACADEMICO": ["2022-23"] * total,
        "NU_PREFERENCIA": preferencia,
    })
    return df


def make_df_base_con_recordtype(n_grado=5, n_master=3) -> pd.DataFrame:
    """DataFrame con RECORDTYPEID real de Salesforce (IDs confirmados por equipo UNAV)."""
    df = make_df_base(n_grado, n_master)
    df["RECORDTYPEID"] = (
        ["012w0000000K4QPAA0"] * n_grado
        + ["012w0000000K4QUAA0"] * n_master
    )
    df["RECORDTYPENAME"] = (
        ["Solicitud Admisión Grado"] * n_grado
        + ["Solicitud Admisión Máster"] * n_master
    )
    return df


# ─── Tests de split_grado_master ──────────────────────────────────────────────

def test_split_grado_no_contiene_master():
    df = make_df_base_con_recordtype(n_grado=5, n_master=3)
    result = split_grado_master(df, "grado")
    assert len(result) == 5
    assert not result["RECORDTYPEID"].isin({"012w0000000K4QUAA0", "012w0000000K4QQAA0"}).any()


def test_split_master_todos_correctos():
    df = make_df_base_con_recordtype(n_grado=5, n_master=3)
    result = split_grado_master(df, "master")
    assert len(result) == 3
    assert result["RECORDTYPEID"].isin({"012w0000000K4QUAA0", "012w0000000K4QQAA0"}).all()


def test_split_grado_conteo_correcto():
    df = make_df_base_con_recordtype(n_grado=5, n_master=3)
    assert len(split_grado_master(df, "grado")) == 5
    assert len(split_grado_master(df, "master")) == 3


def test_split_ambos_ids_grado():
    """Los dos IDs de grado (K4QPAA0 y K4QTAA0) deben clasificarse como grado."""
    df = make_df_base_con_recordtype(n_grado=0, n_master=0)
    extra = pd.DataFrame({
        "ID": ["A", "B"], "TITULACION": ["x", "x"],
        "RECORDTYPEID": ["012w0000000K4QPAA0", "012w0000000K4QTAA0"],
    })
    df = pd.concat([df, extra], ignore_index=True).fillna(0)
    assert len(split_grado_master(df, "grado")) == 2


def test_split_ambos_ids_master():
    """Los dos IDs de máster (K4QUAA0 y K4QQAA0) deben clasificarse como master."""
    df = make_df_base_con_recordtype(n_grado=0, n_master=0)
    extra = pd.DataFrame({
        "ID": ["A", "B"], "TITULACION": ["x", "x"],
        "RECORDTYPEID": ["012w0000000K4QUAA0", "012w0000000K4QQAA0"],
    })
    df = pd.concat([df, extra], ignore_index=True).fillna(0)
    assert len(split_grado_master(df, "master")) == 2


def test_split_recordtypeid_priorizado_sobre_recordtypename():
    """RECORDTYPEID debe tener prioridad aunque RECORDTYPENAME diga otra cosa."""
    df = make_df_base_con_recordtype(n_grado=5, n_master=3)
    df["RECORDTYPENAME"] = "Solicitud Admisión Grado"  # todos dicen grado
    assert len(split_grado_master(df, "master")) == 3, "RECORDTYPEID debe mandar"


def test_split_fallback_recordtypename_sin_id():
    """Sin RECORDTYPEID, usa RECORDTYPENAME con subcadena 'ster'."""
    df = make_df_base(n_grado=5, n_master=3)
    df["RECORDTYPENAME"] = (
        ["Solicitud Admisión Grado"] * 5
        + ["Solicitud Admisión Máster"] * 3
    )
    assert len(split_grado_master(df, "master")) == 3


def test_split_fallback_titulacion_sin_id_ni_nombre():
    """Sin RECORDTYPEID ni RECORDTYPENAME, usa TITULACION con 'MASTER'."""
    df = make_df_base(n_grado=5, n_master=3)
    assert len(split_grado_master(df, "master")) == 3


def test_split_sin_ninguna_columna_clasificadora_lanza_error():
    """Sin ninguna columna clasificadora debe lanzar ValueError."""
    df = make_df_base().drop(columns=["TITULACION"])
    with pytest.raises(ValueError, match="TITULACION"):
        split_grado_master(df, "grado")


def test_split_master_vacio_logea_warning(caplog):
    """Si no hay registros de máster, el segmento es vacío y se emite warning."""
    import logging
    df = make_df_base_con_recordtype(n_grado=5, n_master=0)
    with caplog.at_level(logging.WARNING):
        result = split_grado_master(df, "master")
    assert len(result) == 0
    assert any("vacío" in r.message for r in caplog.records)


# ─── Tests de calcular_orden_automatico ───────────────────────────────────────

def test_calcular_orden_automatico_crea_columnas():
    df = make_df_base()
    result, mapa = calcular_orden_automatico(df)
    assert "etapa_compuesta" in result.columns
    assert "etapa_ordinal_num" in result.columns


def test_calcular_orden_automatico_devuelve_dict():
    df = make_df_base()
    _, mapa = calcular_orden_automatico(df)
    assert isinstance(mapa, dict)
    assert len(mapa) > 0


def test_etapa_ordinal_sin_nulos_despues_de_mapeo():
    df = make_df_base()
    result, _ = calcular_orden_automatico(df)
    # Puede haber NaN si hay etapas que no aparecen en el mapa,
    # pero en datos sintéticos bien formados no debería haber
    assert result["etapa_compuesta"].notna().all()


# ─── Tests de imputar ─────────────────────────────────────────────────────────

def test_imputar_no_nans_en_vars_cero_logico():
    df = make_df_base()
    result = imputar(df)
    for col in VARS_CERO_LOGICO:
        if col in result.columns:
            assert result[col].isna().sum() == 0, \
                f"Quedan NaN en {col} tras imputación"


def test_imputar_renta_familiar_sin_nulos():
    df = make_df_base()
    result = imputar(df)
    if "FO_rentaFam_ges__c" in result.columns:
        assert result["FO_rentaFam_ges__c"].isna().sum() == 0


def test_imputar_beca_sin_nulos():
    df = make_df_base()
    result = imputar(df)
    if "CH_MATRICULA_SUJETA_BECA" in result.columns:
        assert result["CH_MATRICULA_SUJETA_BECA"].isna().sum() == 0


def test_imputar_vars_cero_logico_rellenadas_con_cero():
    df = make_df_base()
    # Forzar NaN en una columna
    col = "NU_NOTA_MEDIA_ADMISION"
    if col in df.columns:
        df[col] = np.nan
        result = imputar(df)
        assert (result[col] == 0).all()


# ─── Tests de crear_vinculacion_previa ────────────────────────────────────────

def test_vinculacion_previa_existe():
    df = make_df_base()
    result = crear_vinculacion_previa(df)
    assert "vinculacion_previa" in result.columns


def test_vinculacion_previa_es_binaria():
    df = make_df_base()
    result = crear_vinculacion_previa(df)
    valores = result["vinculacion_previa"].unique()
    assert set(valores).issubset({0, 1}), \
        f"vinculacion_previa contiene valores no binarios: {valores}"


def test_vinculacion_previa_sin_cols_asigna_cero():
    df = make_df_base().drop(columns=[c for c in COLS_VINCULACION if c in make_df_base().columns])
    result = crear_vinculacion_previa(df)
    assert (result["vinculacion_previa"] == 0).all()


# ─── Tests de get_safe_cols ───────────────────────────────────────────────────

def test_safe_cols_sin_columnas_object():
    df = make_df_base()
    df, _ = calcular_orden_automatico(df)
    df = imputar(df)
    df = crear_vinculacion_previa(df)
    df = df.drop(columns=[c for c in COLS_ID if c in df.columns])
    if "target" in df.columns:
        df_feats = df.drop(columns=["target"])
    else:
        df_feats = df

    safe = get_safe_cols(df_feats)
    object_in_safe = df_feats[safe].select_dtypes(include="object").columns.tolist()
    assert len(object_in_safe) == 0, \
        f"Hay columnas object en safe_cols: {object_in_safe}"


def test_safe_cols_sin_pca():
    df = make_df_base()
    df["PC1"] = 0.5
    df["PCA1"] = 0.3
    safe = get_safe_cols(df)
    for col in safe:
        assert not col.upper().startswith("PC"), \
            f"Columna PCA en safe_cols: {col}"


def test_safe_cols_excluye_vars_excluir():
    df = make_df_base()
    safe = get_safe_cols(df)
    for col in VARS_EXCLUIR:
        assert col not in safe, \
            f"Variable excluida '{col}' está en safe_cols"


# ─── Tests de preprocess (pipeline completo) ──────────────────────────────────

def test_preprocess_grado_devuelve_tres_elementos():
    df = make_df_base(n_grado=5, n_master=3)
    result = preprocess(df, "grado")
    assert len(result) == 3


def test_preprocess_grado_shape_correcto():
    df = make_df_base(n_grado=5, n_master=3)
    df_model, safe_cols, df_ids = preprocess(df, "grado")
    assert df_model.shape[0] == 5, "El DataFrame de grado debe tener 5 filas"
    assert df_model.shape[1] == len(safe_cols)


def test_preprocess_master_shape_correcto():
    df = make_df_base(n_grado=5, n_master=3)
    df_model, safe_cols, df_ids = preprocess(df, "master")
    assert df_model.shape[0] == 3, "El DataFrame de master debe tener 3 filas"


def test_preprocess_df_model_sin_target():
    df = make_df_base()
    df_model, safe_cols, _ = preprocess(df, "grado")
    assert "target" not in df_model.columns


def test_preprocess_df_ids_contiene_id():
    df = make_df_base()
    _, _, df_ids = preprocess(df, "grado")
    assert "ID" in df_ids.columns


def test_preprocess_df_ids_contiene_target_si_existe():
    df = make_df_base()
    _, _, df_ids = preprocess(df, "grado")
    assert "target" in df_ids.columns


def test_preprocess_sin_nulos_en_modelo():
    df = make_df_base()
    df_model, _, _ = preprocess(df, "grado")
    nulos = df_model.isna().sum().sum()
    assert nulos == 0, f"Quedan {nulos} NaN en df_model tras preprocesado"
