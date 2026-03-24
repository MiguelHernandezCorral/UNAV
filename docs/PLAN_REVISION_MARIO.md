# Plan de Revisión — Resultados bajos en predicciones 2026/2027

**Para:** Mario (Data Scientist)
**Contexto:** El modelo se entrenó con datos hasta 2024/2025 y se testeó con 2025/2026 obteniendo ~80% de accuracy. Al aplicarlo sobre datos reales de 2026/2027, los resultados del notebook de análisis muestran métricas sensiblemente más bajas. Este documento describe las posibles causas y los pasos concretos para diagnosticarlas y corregirlas.

---

## Posibles causas (de más probable a menos probable)

| # | Causa | Impacto esperado |
|---|-------|-----------------|
| A | TARGET_REAL vacío o incompleto — el curso 2026/27 aún no ha cerrado | Métricas falsamente bajas o nulas |
| B | Los candidatos están en etapas tempranas del proceso (poca información) | Probabilidades bajas y poco fiables en todos |
| C | Variables clave ausentes o mal imputadas (nulos → 0 incorrecto) | Degradación sistemática del score |
| D | Drift de distribución — el perfil del candidato 2026/27 es distinto | Degradación progresiva del modelo |
| E | Datos insuficientes en algún segmento (pocas filas en Máster) | Métricas inestables |
| F | Bug en la pipeline de predicción (etapas, mapeos, join) | Error sistemático corregible sin reentrenar |

---

## BLOQUE 1 — Verificar si TARGET_REAL tiene datos

> **Por qué:** Si el curso 2026/27 no ha terminado, los alumnos que acabarán matriculándose aún tienen TARGET_REAL = NULL. Eso hace que la sección de métricas del notebook no tenga datos y los resultados parezcan malos cuando en realidad el modelo simplemente no puede evaluarse todavía.

### Tarea 1.1 — Contar cuántos registros tienen TARGET_REAL

```python
import pandas as pd, sys
sys.path.insert(0, 'src')
from dotenv import load_dotenv; load_dotenv()
from oracle_connector import OracleConnector

conn = OracleConnector()
df = pd.DataFrame(conn.read_table('PMAT_PREDICTION'))
df['TARGET_REAL'] = pd.to_numeric(df['TARGET_REAL'], errors='coerce')

print(f"Total registros         : {len(df):,}")
print(f"Con TARGET_REAL         : {df['TARGET_REAL'].notna().sum():,}")
print(f"Sin TARGET_REAL (null)  : {df['TARGET_REAL'].isna().sum():,}")
print(f"TARGET_REAL = 1 (matr.) : {(df['TARGET_REAL']==1).sum():,}")
print(f"TARGET_REAL = 0 (no m.) : {(df['TARGET_REAL']==0).sum():,}")
```

**Interpretación:**
- Si >90% de TARGET_REAL es NULL → el curso no ha cerrado. Las métricas del notebook no son válidas aún. **El modelo puede estar funcionando correctamente.**
- Si TARGET_REAL tiene datos pero son pocos (<500) → muestra insuficiente para evaluar.
- Si TARGET_REAL tiene bastantes datos → continuar al Bloque 2.

---

## BLOQUE 2 — Analizar en qué etapas están los candidatos

> **Por qué:** El modelo es mucho menos seguro en etapas iniciales (poca información disponible). Si la mayoría de candidatos 2026/27 están en fases tempranas, las probabilidades serán naturalmente bajas — no es un fallo del modelo, es la realidad del proceso.

### Tarea 2.1 — Distribución de etapas actuales

```python
df['etapa_compuesta'] = (
    df['ETAPA'].fillna('NA').astype(str).str.strip() + '__' +
    df['SUBETAPA'].fillna('NA').astype(str).str.strip()
)
resumen_etapas = (
    df.groupby('etapa_compuesta')
    .agg(n=('OPP_ID','count'), prob_media=('PROBABILIDAD', 'mean'))
    .sort_values('n', ascending=False)
    .head(20)
)
print(resumen_etapas.to_string())
```

### Tarea 2.2 — Comparar con la distribución de etapas del dataset de entrenamiento

```python
df_ds = pd.DataFrame(conn.read_table('DATASET_LIMPIO'))
etapas_train = df_ds['PL_ETAPA__C'].value_counts(normalize=True).head(10)
etapas_pred  = df['ETAPA'].value_counts(normalize=True).head(10)

print("=== ENTRENAMIENTO ===")
print(etapas_train.to_string())
print("\n=== PREDICCIONES ACTUALES ===")
print(etapas_pred.to_string())
```

**Interpretación:**
- Si en el entrenamiento el grueso estaba en etapas avanzadas (Matrícula, Admisión) pero ahora están en Solicitud/Interés → las probabilidades serán bajas por diseño.
- Calcular qué % de los candidatos actuales está en las 3 primeras etapas del funnel.

---

## BLOQUE 3 — Verificar calidad de variables en DATASET_LIMPIO

> **Por qué:** Si variables clave llegan con muchos nulos y se imputan a 0, el modelo recibe información distorsionada. En el entrenamiento esas variables tenían valores reales; ahora pueden estar vacías para candidatos nuevos.

### Tarea 3.1 — % de nulos por variable clave

```python
cols_modelo = [
    'CU_IMPORTE_TOTAL', 'NU_MEDIA_EXPEDIENTE_UNIVERSITA',
    'PORCENTAJE_PAGADO_FINAL', 'NU_NOTA_MEDIA_ADMISION',
    'NU_NOTA_MEDIA_1_BACH__PC', 'NU_RESULTADO_ADMISION_PUNTOS',
    'NUM_ASISTENCIAS_ACUM', 'NUM_SOLICITUDES_ACUM',
    'FO_RENTAFAM_GES__C', 'CH_MATRICULA_SUJETA_BECA',
]
pct_nulos = (
    df_ds[cols_modelo]
    .isna().mean()
    .mul(100)
    .sort_values(ascending=False)
    .rename('% nulos')
)
print(pct_nulos.to_string())
```

**Umbrales de alerta:**
- >50% nulos en una variable clave → sospechoso, investigar por qué
- >90% nulos → la variable llega vacía desde SF, la imputación a 0 puede sesgar el modelo

### Tarea 3.2 — Comparar distribuciones: entrenamiento vs. 2026/27

Para cada variable clave, comparar media/desviación entre el histórico y los datos actuales:

```python
# Cargar el CSV de test (o el dataset de entrenamiento si está disponible)
# df_train = pd.read_excel('datos/...')   # ajustar ruta
# Para cada variable:
for col in cols_modelo:
    if col in df_ds.columns:
        media_actual = df_ds[col].mean()
        # media_train = df_train[col].mean()   # descomentar si tienes el CSV
        pct_nulos_col = df_ds[col].isna().mean() * 100
        print(f"{col:<45} media={media_actual:.2f}  nulos={pct_nulos_col:.1f}%")
```

**Señales de alerta:**
- Media muy distinta al histórico → distribución cambiada (drift)
- Media ~0 con muchos nulos → la variable se está imputando incorrectamente

### Tarea 3.3 — Verificar que CU_IMPORTE_TOTAL llega con datos reales

Esta variable fue problemática en el pasado. Verificar:

```python
n_cero    = (df_ds['CU_IMPORTE_TOTAL'] == 0).sum()
n_nulo    = df_ds['CU_IMPORTE_TOTAL'].isna().sum()
n_total   = len(df_ds)
print(f"CU_IMPORTE_TOTAL = 0   : {n_cero:,} ({n_cero/n_total:.1%})")
print(f"CU_IMPORTE_TOTAL = null: {n_nulo:,} ({n_nulo/n_total:.1%})")
print(f"Media (excluyendo 0)   : {df_ds[df_ds['CU_IMPORTE_TOTAL']>0]['CU_IMPORTE_TOTAL'].mean():.2f}")
```

Si >80% son 0 o nulos, hay un problema en la ingesta desde Salesforce.

---

## BLOQUE 4 — Detectar drift de distribución

> **Por qué:** El modelo aprendió patrones de candidatos 2022–2025. Si el perfil del candidato 2026/27 ha cambiado (más internacionales, nuevos programas, cambio en la forma de captar candidatos), el modelo puede degradarse aunque la pipeline sea perfecta.

### Tarea 4.1 — Comparar variables numéricas: distribución histórica vs. actual

```python
import matplotlib.pyplot as plt

# Cargar datos de test del modelo (ajustar ruta)
# df_test = pd.read_excel('datos/.../test_grado_completo_con_scoring.xlsx')

vars_check = ['CU_IMPORTE_TOTAL', 'NUM_ASISTENCIAS_ACUM', 'PORCENTAJE_PAGADO_FINAL']
fig, axes = plt.subplots(1, len(vars_check), figsize=(15, 4))
for ax, col in zip(axes, vars_check):
    if col in df_ds.columns:
        df_ds[col].dropna().hist(bins=30, ax=ax, alpha=0.6, label='2026/27', color='steelblue')
        # df_test[col].dropna().hist(bins=30, ax=ax, alpha=0.6, label='histórico', color='orange')
        ax.set_title(col)
        ax.legend()
plt.tight_layout()
plt.savefig('docs/drift_variables.png')
plt.show()
```

### Tarea 4.2 — Comparar titulaciones: ¿aparecen programas nuevos?

```python
tit_actuales = set(df_ds['TITULACION'].dropna().unique())
# tit_train = set(df_train['TITULACION'].dropna().unique())
# nuevas = tit_actuales - tit_train
# print("Titulaciones nuevas (no vistas en entrenamiento):", nuevas)

print(f"Titulaciones en 2026/27: {len(tit_actuales)}")
print(df_ds['TITULACION'].value_counts().head(15).to_string())
```

---

## BLOQUE 5 — Verificar la pipeline técnica

> **Por qué:** Puede haber un bug silencioso en el preprocesado o en cómo se mapean las variables al modelo.

### Tarea 5.1 — Ejecutar la pipeline en modo dry-run y revisar los logs

```bash
cd /home/infra/jvelazquezc/UNAV   # o la ruta local
source .venv/bin/activate
python src/pipeline.py --phases fase2 fase4 --dry-run
```

Revisar en `logs/pipeline.log` que:
- No aparezcan warnings de columnas añadidas con 0 inesperadamente
- El número de filas procesadas sea el esperado
- No haya errores silenciosos en `preprocessor.py`

### Tarea 5.2 — Revisar el preprocesado manualmente

```python
import sys; sys.path.insert(0, 'src')
from dotenv import load_dotenv; load_dotenv()
from preprocessor import load_dataset_limpio, preprocess
import logging; logging.basicConfig(level=logging.DEBUG)

df_full = load_dataset_limpio()
df_model, safe_cols, df_ids = preprocess(df_full, tipo='grado')

print(f"Filas preprocesadas: {len(df_model):,}")
print(f"Features usadas: {len(safe_cols)}")
print("\nFeatures con >50% ceros:")
pct_cero = (df_model == 0).mean()
print(pct_cero[pct_cero > 0.5].sort_values(ascending=False).to_string())
```

**Señales de alerta:**
- Features con >80% ceros que en entrenamiento tenían distribución real → imputación incorrecta
- `safe_cols` con muchas menos features de las esperadas → puede que se estén filtrando columnas que no deberían

### Tarea 5.3 — Verificar que el modelo recibe exactamente las features que espera

```python
from pycaret.classification import load_model
modelo_grado = load_model('models/modelo_final_grado')

features_modelo = list(modelo_grado.feature_names_in_)
features_actuales = safe_cols

faltan  = [f for f in features_modelo if f not in features_actuales]
sobran  = [f for f in features_actuales if f not in features_modelo]

print(f"Features que espera el modelo : {len(features_modelo)}")
print(f"Features disponibles          : {len(features_actuales)}")
print(f"Faltan (se rellenan con 0)    : {len(faltan)} → {faltan}")
print(f"Sobran (se ignoran)           : {len(sobran)} → {sobran[:5]}")
```

Si hay muchas features que faltan y se rellenan con 0, el modelo está recibiendo datos muy distintos a los del entrenamiento.

---

## BLOQUE 6 — Decisión: ¿reentrenar o no?

Una vez completados los bloques anteriores, seguir este árbol de decisión:

```
¿TARGET_REAL tiene suficientes datos para evaluar (>500 registros)?
    NO  → El modelo no puede evaluarse aún. Esperar al cierre del curso.
          Monitorizar mensualmente.
    SÍ  →
        ¿Accuracy/AUC en datos 2026/27 es >75%?
            SÍ  → El modelo funciona bien. Revisar umbrales o comunicación.
            NO  →
                ¿>30% features llegan con >80% nulos/ceros?
                    SÍ  → Problema de datos. Arreglar ingesta SF → Oracle
                          primero. Reejecutar sin reentrenar.
                    NO  →
                        ¿Hay drift significativo en variables clave?
                            NO  → Bug en pipeline. Revisar preprocessor.py.
                            SÍ  → Reentrenar el modelo con datos hasta 2026
                                  (incluir 2025/2026 en train, 2026/2027 en test)
```

### Tarea 6.1 — Si se decide reentrenar

1. Exportar DATASET_LIMPIO actual a CSV:
```python
df_ds.to_csv('datos/dataset_limpio_2026.csv', index=False)
```

2. Abrir `notebooks/03_Modelado.ipynb` y:
   - Actualizar la ruta de datos al CSV nuevo
   - Cambiar el split temporal: train hasta dic-2025, test desde ene-2026
   - Ejecutar todas las celdas
   - Comparar métricas en test con las del modelo actual

3. Si las métricas mejoran >2 puntos de AUC → guardar el nuevo modelo en `models/` y actualizar la pipeline.

---

## Checklist de entrega para Mario

Cuando termines la revisión, reporta:

- [ ] Bloque 1: ¿Cuántos registros tienen TARGET_REAL? ¿Es suficiente para evaluar?
- [ ] Bloque 2: ¿En qué etapas están la mayoría de candidatos? ¿Son etapas tempranas?
- [ ] Bloque 3: ¿Qué variables tienen >50% nulos? ¿CU_IMPORTE_TOTAL llega correctamente?
- [ ] Bloque 4: ¿Hay variables con distribución claramente distinta al histórico?
- [ ] Bloque 5: ¿Cuántas features del modelo se están rellenando con 0?
- [ ] Decisión final: ¿es un problema de datos, de pipeline o hay que reentrenar?

---

## Comandos de referencia rápida

```bash
# Conectarse a la MV
ssh jvelazquezc@hydra4-pre.unav.es

# Activar entorno
source /home/infra/jvelazquezc/UNAV/.venv/bin/activate

# Ver últimas predicciones en Oracle
python -c "
import sys; sys.path.insert(0, 'src')
from dotenv import load_dotenv; load_dotenv()
from oracle_connector import OracleConnector
import pandas as pd
conn = OracleConnector()
df = pd.DataFrame(conn.read_table('PMAT_PREDICTION'))
print('Total:', len(df))
print('Última predicción:', df['FECHA_PRED'].max())
print('Con TARGET_REAL:', df['TARGET_REAL'].notna().sum())
"

# Lanzar solo limpieza + predicciones
python src/pipeline.py --phases fase2 fase4

# Ver logs en tiempo real
tail -f logs/pipeline.log
```
