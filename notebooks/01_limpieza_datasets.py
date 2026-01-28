# ============================================================
# SCRIPT · DATASET DE TRATAMIENTO DEFINITIVO FINAL
# ============================================================
# Objetivo:
#   - Construir el dataset final de modelización a partir de Salesforce
#   - Integrar información académica, económica, actividades y tiempos
#   - Controlar leakage de información futura
#   - Dejar el dataset listo para análisis y ML
# ============================================================

import pandas as pd
import numpy as np
from utils import crear_target, eliminar_columnas_na, calcular_tiempos_etapas, integrar_actividades_progresivo_por_curso

# Funciones auxiliares definidas en utils.py
# - crear_target: construye la variable objetivo a partir del historial de etapas
# - eliminar_columnas_na: elimina columnas con exceso de valores nulos
# - calcular_tiempos_etapas: calcula duración en cada etapa del funnel
# - integrar_actividades_progresivo_por_curso: agrega actividades acumuladas

# ============================================================
# 1️⃣ CARGA DE DATOS
# ============================================================
# Se carga el Excel completo de Salesforce
# Cada hoja corresponde a una entidad distinta
# ============================================================

ruta_excel = r"..\datos\01. Datos originales\DataSET_SF - V2.xlsx"
dfs = pd.read_excel(ruta_excel, sheet_name=None)

# Asignar cada hoja a un dataframe independiente
# El orden debe coincidir con el Excel original
oportunidad = list(dfs.values())[0]
cuenta = list(dfs.values())[1]
ecb = list(dfs.values())[2]
solicitud_ban = list(dfs.values())[3]
casos = list(dfs.values())[4]
correos = list(dfs.values())[5]
historial_actividad = list(dfs.values())[6]
historial_etapas = list(dfs.values())[7]

# ============================================================
# 2️⃣ LIMPIEZA INICIAL DE NAS Y COLUMNAS
# ============================================================
# Se eliminan columnas con un porcentaje de NA superior al umbral
# Esto reduce ruido y dimensionalidad desde el inicio
# ============================================================

def eliminar_columnas_na(df, umbral=0.9):
    """Elimina columnas con más de un umbral de valores NA"""
    return df.loc[:, df.isna().mean() < umbral]


# Limpieza genérica (no modifica los dataframes originales)
for df in [oportunidad, cuenta, ecb, solicitud_ban, casos, correos, historial_actividad, historial_etapas]:
    df = eliminar_columnas_na(df)


# Limpieza efectiva sobre los dataframes clave
oportunidad = eliminar_columnas_na(oportunidad)
cuenta = eliminar_columnas_na(cuenta)
ecb = eliminar_columnas_na(ecb)

# ============================================================
# 3️⃣ CREACIÓN DEL TARGET
# ============================================================
# Se construye la variable objetivo (target) usando el historial de etapas
# ============================================================

oportunidad = crear_target(oportunidad, historial_etapas)


# Unión de oportunidad con datos de cuenta/persona
# Se hace LEFT JOIN para no perder oportunidades

df_unido = pd.merge(
    oportunidad, 
    cuenta, 
    left_on='ACCOUNTID', 
    right_on='ID18', 
    how='left',
    suffixes=('', '_cuenta')
)


# ============================================================
# 4️⃣ CONSTRUCCIÓN VARIABLES DERIVADAS
# ============================================================
# Se crean variables explicativas a partir de campos originales
# ============================================================

# Normalización del plazo de admisión
# Se agrupan valores heterogéneos en categorías consistentes
def normalizar_plazo(x):
    if pd.isna(x): return "Rolling"
    x = str(x).strip().lower()
    if "dic" in x: return "Diciembre"
    if "mar" in x: return "Marzo"
    return "Otros"

df_unido['PLAZO_ADMISION_LIMPIO'] = df_unido['PL_PLAZO_ADMISION'].apply(normalizar_plazo)

# Unión con información económica (ECB)
# Se incorporan precios y renta familiar
ecb_vars = ['LK_oportunidad__c', 'FO_rentaFam_ges__c', 'CU_precioOrdinario_def__c', 'CU_precioAplicado_def__c']
df_definitivo = pd.merge(
    df_unido,
    ecb[ecb_vars],
    left_on='ID',
    right_on='LK_oportunidad__c',
    how='left'
)


# Cálculo del porcentaje pagado final
# Se controla división por cero
df_definitivo['PORCENTAJE_PAGADO_FINAL'] = (
    df_definitivo['CU_precioAplicado_def__c'] / df_definitivo['CU_precioOrdinario_def__c'] * 100
)
df_definitivo.loc[df_definitivo['CU_precioOrdinario_def__c'] <= 0, 'PORCENTAJE_PAGADO_FINAL'] = np.nan


# Guardado intermedio (dataset de análisis)
ruta_salida = r"..\datos\01. Datos originales\dataset_analisis_final.csv"
df_definitivo.to_csv(ruta_salida, sep=";", index=False)


# ============================================================
# 5️⃣ TIEMPO EN CADA ETAPA
# ============================================================
# Se calcula el tiempo pasado en cada etapa del funnel
# ============================================================

historial_etapas_tiempo = calcular_tiempos_etapas(historial_etapas)
df_definitivo = historial_etapas_tiempo.merge(df_definitivo, left_on='LK_Oportunidad__c', right_on='ID', how='left')

# ============================================================
# 6️⃣ HISTORIAL DE ACTIVIDADES
# ============================================================
# Se integran actividades acumuladas por curso
# Evita usar información futura respecto a la etapa
# ============================================================

df_definitivo = integrar_actividades_progresivo_por_curso(df_definitivo, historial_actividad)

# ============================================================
# 7️⃣ CONTROL DE INFORMACIÓN FUTURA (LEAKAGE)
# ============================================================
# Se eliminan variables económicas si aparecen en etapas tempranas
# ============================================================

etapas_pago = ['Solicitud', 'Pruebas', 'Admisión académica']
vars_pago = ['PAID_AMOUNT','MINIMUMPAYMENTPAYED','CU_precioAplicado_def__c','PORCENTAJE_PAGADO_FINAL']
vars_pago = [v for v in vars_pago if v in df_definitivo.columns]

mask_futuro = (df_definitivo['PL_Etapa__c'].isin(etapas_pago)) & (df_definitivo[vars_pago].notna().any(axis=1))
df_definitivo.loc[mask_futuro, vars_pago] = np.nan

# ============================================================
# 8️⃣ SELECCIÓN VARIABLES FINALES
# ============================================================
# Se define explícitamente el conjunto final de variables
# ============================================================
columnas_seleccionadas = [
    'ACCOUNTID', 'ID','ID18__PC', 'target', 'desmatriculado', 'PL_CURSO_ACADEMICO', 'CH_NACIONAL',
    'NU_NOTA_MEDIA_ADMISION', 'NU_NOTA_MEDIA_1_BACH__PC', 'CH_PRUEBAS_CALIFICADAS', 
    'NU_RESULTADO_ADMISION_PUNTOS', 'PL_RESOLUCION_DEFINITIVA', 'TITULACION', 'CENTROENSENANZA',
    'MINIMUMPAYMENTPAYED', 'PAID_AMOUNT', 'PAID_PERCENT', 'CH_PAGO_SUPERIOR', 
    'CH_MATRICULA_SUJETA_BECA', 'CH_AYUDA_FINANCIACION', 'CU_IMPORTE_TOTAL',
    'CH_VISITACAMPUS__PC', 'CH_ENTREVISTA_PERSONAL__PC', 'ACC_DTT_FECHAULTIMAACTIVIDAD', 
    'NU_PREFERENCIA', 'STAGENAME', 'PL_SUBETAPA',
    'CH_HIJO_EMPLEADO__PC', 'CH_HIJO_PROFESOR_ASOCIADO__PC', 'CH_HERMANOS_ESTUDIANDO_UNAV__P', 
    'CH_HIJO_MEDICO__PC', 'YEARPERSONBIRTHDATE', 'NAMEX', 'CH_FAMILIA_NUMEROSA__PC', 
    'PL_SITUACION_SOCIO_ECONOMICA', 'LEADSOURCE', 'PL_ORIGEN_DE_SOLICITUD', 
    'PL_PLAZO_ADMISION', 'RECORDTYPENAME','PLAZO_ADMISION_LIMPIO','FO_rentaFam_ges__c','CU_precioOrdinario_def__c',
    'CU_precioAplicado_def__c','PORCENTAJE_PAGADO_FINAL','tiempo_etapa_dias','tiempo_entre_etapas_dias','num_asistencias_acum', 'num_solicitudes_acum'
]


#columnas_finales = [c for c in columnas_finales if c in df_definitivo.columns]
df_definitivo = df_definitivo[columnas_seleccionadas]

# ============================================================
# 9️⃣ GUARDAR DATASET TRATAMIENTO DEFINITIVO
# ============================================================

ruta_salida = r"..\datos\01. Datos originales\dataset_tratamiento_final.csv"
df_definitivo.to_csv(ruta_salida, sep=";", index=False)

print(f"✅ Dataset de tratamiento definitivo guardado en: {ruta_salida}")
print(f"Dimensiones: {df_definitivo.shape}")
df_definitivo.head()