import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
def analisis_na_por_columna(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe un DataFrame y devuelve un DataFrame con:
    - Nombre de la columna
    - Total de valores NA
    - Porcentaje de valores NA
    """
    num_registros = len(df)
    total_na_col = df.isna().sum()
    porcentaje_na_col = (total_na_col / num_registros) * 100
    
    resumen_df = pd.DataFrame({
        "columna": df.columns,
        "total_na": total_na_col.values,
        "porcentaje_na": porcentaje_na_col.values
    })
    resumen_df = resumen_df.sort_values(by="porcentaje_na", ascending=False).reset_index(drop=True)
    return resumen_df

# Ejemplo de uso:
# df = pd.DataFrame({"A": [1, None, 3], "B": [4, 5, None]})
# print(analisis_na_por_columna(df))

def eliminar_columnas_na(df: pd.DataFrame, umbral: float = 90) -> pd.DataFrame:
    """
    Elimina columnas que tengan más de 'umbral'% de valores NA.
    
    Parámetros:
    df: DataFrame de entrada
    umbral: porcentaje máximo permitido de NAs (por defecto 90)
    
    Retorna:
    DataFrame sin las columnas que superen el umbral de NAs
    """
    num_registros = len(df)
    porcentaje_na_col = (df.isna().sum() / num_registros) * 100
    
    columnas_a_eliminar = porcentaje_na_col[porcentaje_na_col > umbral].index
    df_filtrado = df.drop(columns=columnas_a_eliminar)
    
    return df_filtrado

# Ejemplo de uso:
# df = pd.DataFrame({"A": [None]*9 + [1], "B": [4, 5, None, 7, 8, None, 10, None, None, None]})
# print(eliminar_columnas_na(df))

def crear_target(oportunidad: pd.DataFrame, historial_etapas: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la columna 'target' para cada oportunidad según la lógica:
    - Existe etapa 'Matrícula OOGG' con estado 'formalizada'
    - No existe etapa 'Desmatriculado'
    """
    
    # Filtrar historial por Matrícula OOGG y estado formalizada
    matricula_formalizada = historial_etapas[
        (historial_etapas['PL_Etapa__c'] == 'Matrícula OOGG') &
        (historial_etapas['PL_Subetapa__c'] == 'Formalizada')
    ]['LK_Oportunidad__c'].unique()
    print('Hay un total de '+str(len(matricula_formalizada))+' matrículas formalizadas. Un '+str(round(len(matricula_formalizada)/len(historial_etapas['LK_Oportunidad__c'].unique())*100,2))+'% del total de oportunidades')

    
    
    # Filtrar historial por Desmatriculado
    desmatriculado = historial_etapas[
        historial_etapas['PL_Subetapa__c'] == 'Desmatriculado'
    ]['LK_Oportunidad__c'].unique()
    print('Hay un total de '+str(len(desmatriculado))+' desmatriculados. Un '+str(round(len(desmatriculado)/len(matricula_formalizada)*100,2))+'% del total de matriculados')
    # Crear target: 1 si está en matricula formalizada y no en desmatriculado
    oportunidad['target'] = oportunidad['ID'].apply(
        lambda x: 1 if (x in matricula_formalizada and x not in desmatriculado) else 0
    )
    
    return oportunidad
import pandas as pd

def crear_target_auditado(oportunidad: pd.DataFrame, historial_etapas: pd.DataFrame) -> pd.DataFrame:
    print("--- INICIANDO AUDITORÍA DE INTEGRIDAD ---")
    
    # 1. Análisis de Intersección (Venn Diagram Logic)
    ids_oportunidad = set(oportunidad['ID'].unique())
    ids_historial = set(historial_etapas['LK_Oportunidad__c'].unique())
    
    comunes = ids_oportunidad.intersection(ids_historial)
    solo_oportunidad = ids_oportunidad - ids_historial
    solo_historial = ids_historial - ids_oportunidad
    
    print(f"Total IDs en Maestro Oportunidades: {len(ids_oportunidad)}")
    print(f"Total IDs en Historial Etapas: {len(ids_historial)}")
    print(f"✅ Coincidencias exactas: {len(comunes)}")
    print(f"⚠️ IDs en Maestro pero SIN historial: {len(solo_oportunidad)} (Posibles registros huérfanos)")
    print(f"⚠️ IDs en Historial pero NO en Maestro: {len(solo_historial)} (Oportunidades eliminadas o filtradas)")

    # 2. Detección de Duplicados
    dups_master = oportunidad['ID'].duplicated().sum()
    if dups_master > 0:
        print(f"❌ ¡ATENCIÓN!: Tienes {dups_master} duplicados en el DataFrame Maestro.")

    # 3. Lógica de Target (Optimizada con Isin para velocidad)
    matricula_formalizada = set(historial_etapas[
        (historial_etapas['PL_Etapa__c'] == 'Matrícula OOGG') &
        (historial_etapas['PL_Subetapa__c'] == 'Formalizada')
    ]['LK_Oportunidad__c'].unique())
    
    desmatriculado = set(historial_etapas[
        historial_etapas['PL_Subetapa__c'] == 'Desmatriculado'
    ]['LK_Oportunidad__c'].unique())
    
    # Calculamos el target usando conjuntos (sets), que es mucho más rápido que .apply(lambda)
    # Target = 1 si está en formalizados Y NO en desmatriculados
    ids_target_1 = matricula_formalizada - desmatriculado
    
    oportunidad['target'] = oportunidad['ID'].isin(ids_target_1).astype(int)
    
    # 4. Resumen Final
    total_matriculas = oportunidad['target'].sum()
    print(f"\n--- RESUMEN TARGET ---")
    print(f"Matrículas Finales (Target=1): {total_matriculas}")
    print(f"Tasa de Conversión Total: {round(total_matriculas / len(oportunidad) * 100, 2)}%")
    
    return oportunidad



def calcular_tiempos_etapas(historial_etapas: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la duración de cada etapa y el tiempo de transición, 
    sustituyendo cualquier valor nulo (NaN/NA) por cero.
    """
    # 1. Convertir fechas a tipo datetime
    historial_etapas['CreatedDate'] = pd.to_datetime(historial_etapas['CreatedDate'])
    historial_etapas['Fecha_fin_etapa__c'] = pd.to_datetime(historial_etapas['Fecha_fin_etapa__c'])
    
    # 2. Ordenar por oportunidad y fecha de inicio para asegurar el cálculo secuencial
    historial_etapas = historial_etapas.sort_values(by=['LK_Oportunidad__c', 'CreatedDate'])
    
    # 3. Calcular tiempo que tarda en terminar la etapa (Diferencia interna)
    # Rellenamos nulos con 0 en caso de que falte la fecha de fin
    historial_etapas['tiempo_etapa_dias'] = (
        historial_etapas['Fecha_fin_etapa__c'] - historial_etapas['CreatedDate']
    ).dt.days.fillna(0).astype(int)
    
    # 4. Calcular tiempo entre inicio de una etapa y la siguiente (Diferencia secuencial)
    # Usamos shift(-1) para traer la fecha de inicio de la siguiente etapa
    fecha_siguiente = historial_etapas.groupby('LK_Oportunidad__c')['CreatedDate'].shift(-1)
    
    historial_etapas['tiempo_entre_etapas_dias'] = (
        fecha_siguiente - historial_etapas['CreatedDate']
    ).dt.days.fillna(0).astype(int)
    
    return historial_etapas



def limpiar_historial_por_hitos(df_historial, df_principal):
    # 1. Asegurar formato datetime
    df_historial['CreatedDate'] = pd.to_datetime(df_historial['CreatedDate'])
    
    # 2. Obtener la fecha del Hito Académico (Pruebas calificadas)
    # Filtramos por los valores exactos de etapa y subetapa
    hito_acad = df_historial[
        (df_historial['PL_Etapa__c'] == 'Pruebas de admisión') & 
        (df_historial['PL_Subetapa__c'] == 'Pruebas calificadas')
    ].groupby('LK_Oportunidad__c')['CreatedDate'].min().reset_index()
    hito_acad.columns = ['LK_Oportunidad__c', 'fecha_pruebas_calificadas']
    
    # 3. Obtener la fecha del Hito Económico (Matrícula iniciada)
    hito_econ = df_historial[
        (df_historial['PL_Etapa__c'] == 'Matrícula admisión') & 
        (df_historial['PL_Subetapa__c'] == 'Pago Mínimo')
    ].groupby('LK_Oportunidad__c')['CreatedDate'].min().reset_index()
    hito_econ.columns = ['LK_Oportunidad__c', 'fecha_matricula_iniciada']
    
    # 4. Merge: Unir el historial con las fechas de sus propios hitos
    df_merge = pd.merge(df_historial, hito_acad, on='LK_Oportunidad__c', how='left')
    df_merge = pd.merge(df_merge, hito_econ, on='LK_Oportunidad__c', how='left')
    
    # 5. Merge: Traer las columnas del df_principal (académicas y económicas)
    # Unimos por ID de oportunidad
    df_final = pd.merge(df_merge, df_principal, left_on='LK_Oportunidad__c', right_on='ID', how='left')
    
    # 6. Definición de grupos de columnas
    cols_academicas = [
        'NU_NOTA_MEDIA_ADMISION', 'CH_PRUEBAS_CALIFICADAS', 
        'NU_RESULTADO_ADMISION_PUNTOS', 'PL_RESOLUCION_DEFINITIVA'
    ]
    cols_economicas = [
        'MINIMUMPAYMENTPAYED', 'PAID_AMOUNT', 'PAID_PERCENT', 'CH_PAGO_SUPERIOR', 
        'CH_MATRICULA_SUJETA_BECA', 'CH_AYUDA_FINANCIACION', 'CU_IMPORTE_TOTAL'
    ]
    
    # 7. Aplicación de la lógica temporal
    # Lógica Académica: Si la fecha del registro es anterior a las pruebas calificadas (o nunca ocurrieron)
    mask_acad = (df_final['fecha_pruebas_calificadas'].isna()) | (df_final['CreatedDate'] < df_final['fecha_pruebas_calificadas'])
    df_final.loc[mask_acad, cols_academicas] = np.nan
    
    # Lógica Económica: Si la fecha del registro es anterior a la matrícula iniciada (o nunca ocurrió)
    mask_econ = (df_final['fecha_matricula_iniciada'].isna()) | (df_final['CreatedDate'] < df_final['fecha_matricula_iniciada'])
    df_final.loc[mask_econ, cols_economicas] = np.nan
    
    # Limpieza de columnas auxiliares de fecha
    #df_final = df_final .drop(columns=['fecha_pruebas_calificadas', 'fecha_matricula_iniciada'])
    
    return df_final

# Uso del código:
# historial_enriquecido = limpiar_historial_por_hitos(historial_etapas, oportunidad_filtrada)


def integrar_actividades_progresivo_por_curso(df_master, df_actividades):
    print(f"Procesando {len(df_master)} filas con lógica de curso y progresión temporal...")

    # 1. Limpieza de Actividades
    keywords_excluir = ['mail', 'email', 'whatsapp', 'masivo', 'comunicación', 'envío']
    patron = '|'.join(keywords_excluir)
    
    col_tipo_act = 'Campaign.LK_tipoActividadPromocion__r.Name'
    
    # Filtro: No nulos, no vacíos y sin keywords masivas
    df_act_clean = df_actividades[
        df_actividades[col_tipo_act].notna() & 
        (df_actividades[col_tipo_act].str.strip() != '') &
        ~df_actividades[col_tipo_act].str.contains(patron, case=False, na=False)
    ].copy()

    # Preparación de fechas y normalización
    df_act_clean['ActivityDate'] = pd.to_datetime(df_act_clean['CreatedDate'])
    df_master['MasterDate'] = pd.to_datetime(df_master['CreatedDate'])
    df_act_clean['status_lower'] = df_act_clean['Estado_del_miembro__c'].str.lower()

    # 2. Clave única por fila para garantizar la progresión individual
    df_master = df_master.reset_index().rename(columns={'index': 'fila_id_unico'})

    # 3. Join por Contacto Y Curso Académico
    # Maestro: PL_CURSO_ACADEMICO | Actividades: Campaign.AcademicCourse__c
    print("Cruzando datos por ID18__PC y Curso Académico...")
    df_joined = pd.merge(
        df_master[['fila_id_unico', 'ID18__PC', 'MasterDate', 'PL_CURSO_ACADEMICO']], 
        df_act_clean[['ContactId', 'ActivityDate', 'status_lower', 'Campaign.AcademicCourse__c']], 
        left_on=['ID18__PC', 'PL_CURSO_ACADEMICO'],
        right_on=['ContactId', 'Campaign.AcademicCourse__c'],
        how='inner'
    )

    # 4. FILTRO TEMPORAL ESTRICTO
    # Solo actividades ocurridas antes de la fecha de la fila actual
    print("Aplicando filtro temporal progresivo...")
    df_joined = df_joined[df_joined['ActivityDate'] < df_joined['MasterDate']]

    # 5. Contadores vectorizados
    df_joined['es_asiste'] = (df_joined['status_lower'] == 'asiste').astype(int)
    df_joined['es_solicitado'] = (df_joined['status_lower'].isin(['solicitado', 'solicita asistir'])).astype(int)

    # 6. Agrupación por la fila única
    print("Agrupando resultados...")
    resumen = df_joined.groupby('fila_id_unico').agg(
        num_asistencias_acum=('es_asiste', 'sum'),
        num_solicitudes_acum=('es_solicitado', 'sum')
    ).reset_index()

    # 7. Consolidación final
    print("Consolidando en el DataFrame maestro...")
    df_final_v3 = pd.merge(df_master, resumen, on='fila_id_unico', how='left')

    # Rellenar nulos con 0 y limpiar columnas auxiliares
    df_final_v3[['num_asistencias_acum', 'num_solicitudes_acum']] = \
        df_final_v3[['num_asistencias_acum', 'num_solicitudes_acum']].fillna(0).astype(int)
    
    # Eliminamos las columnas de apoyo pero mantenemos el orden original
    resultado = df_final_v3.drop(columns=['fila_id_unico', 'MasterDate'])
    
    print("✅ Proceso completado.")
    return resultado
    
import matplotlib.pyplot as plt

def graficar_top_por_acceso(df, top_n=5):
    # 1. Agrupar y contar oportunidades únicas
    df_counts = df.groupby(['PL_ORIGEN_DE_SOLICITUD', 'TITULACION_DEF', 'target'])['ID'].nunique().reset_index()
    df_counts.columns = ['Acceso', 'Titulación', 'Target', 'Oportunidades']

    # 2. Obtener los tipos de acceso únicos
    accesos = df_counts['Acceso'].unique()
    
    # 3. Crear una figura con subplots (uno por cada tipo de acceso)
    fig, axes = plt.subplots(len(accesos), 1, figsize=(12, 6 * len(accesos)))
    if len(accesos) == 1: axes = [axes] # Manejo de caso con un solo acceso

    for i, acceso in enumerate(accesos):
        # Filtrar datos por acceso y coger las N titulaciones con más volumen total
        data_acceso = df_counts[df_counts['Acceso'] == acceso]
        top_titulaciones = data_acceso.groupby('Titulación')['Oportunidades'].sum().nlargest(top_n).index
        data_top = data_acceso[data_acceso['Titulación'].isin(top_titulaciones)]

        # Pintar en el subplot correspondiente
        sns.barplot(
            ax=axes[i],
            data=data_top,
            y='Titulación',
            x='Oportunidades',
            hue='Target',
            palette={0: '#e74c3c', 1: '#2ecc71'} # Rojo para No, Verde para Sí
        )
        axes[i].set_title(f'Top {top_n} Titulaciones en Acceso: {acceso}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Número de Oportunidades Únicas')
        axes[i].legend(title='Matriculado', labels=['No (0)', 'Sí (1)'])

    plt.tight_layout()
    plt.show()
import pandas as pd

