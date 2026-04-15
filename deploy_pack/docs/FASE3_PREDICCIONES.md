# Fase 3 · Predicciones con Modelos Preentrenados

## Descripción general

Esta fase usa los modelos PyCaret preentrenados (`modelo_final_grado.pkl` y
`modelo_final_master.pkl`) para generar predicciones de probabilidad de matrícula
sobre el dataset limpio almacenado en Oracle (`DATASET_LIMPIO`).

El resultado se almacena en la tabla Oracle `PMAT_PREDICTION` mediante **UPSERT inteligente**:
solo actualiza un registro cuando cambia la probabilidad de matrícula.

---

## Módulos

| Archivo | Rol |
|---|---|
| `models/modelo_final_grado.pkl` | Pipeline PyCaret entrenado para candidatos de Grado |
| `models/modelo_final_master.pkl` | Pipeline PyCaret entrenado para candidatos de Máster |
| `src/preprocessor.py` | Preprocesado pre-PyCaret (feature engineering + imputación) |
| `src/predictor.py` | Carga modelos, lanza predicciones, guarda en Oracle |

---

## Modelos PKL

Los modelos fueron entrenados con PyCaret (`save_model()`) e incluyen el pipeline
completo de normalización y codificación configurado en el `setup()` original.

**Origen:** `notebooks/03_Modelado.ipynb`
**Ubicación en repo:** `models/`
**Criterio de reentrenamiento:** Manual — borrar el PKL correspondiente para forzar reentrenamiento.

### Métricas de referencia (set de test)

| Métrica | Grado | Máster |
|---|---|---|
| AUC-ROC | 0.917 | 0.897 |
| Accuracy | 83.3% | 88.0% |
| Recall (clase 1) | 86.1% | 95.3% |
| Especificidad (clase 0) | 80.9% | 77.6% |

---

## Preprocesado (src/preprocessor.py)

El preprocesado replica exactamente la lógica del notebook `03_Modelado.ipynb`
**antes** de llamar a `predict_model()`.
PyCaret maneja internamente la normalización; aquí se aplican las transformaciones
de negocio:

### Pasos

1. **Carga** → `OracleConnector.read_table('DATASET_LIMPIO')`
2. **Separación Grado/Máster** → por `TITULACION` (contiene/no contiene `'MASTER'`)
3. **Etapa ordinal automática**:
   - `etapa_compuesta` = `PL_Etapa__c` + `"__"` + `PL_Subetapa__c`
   - `etapa_ordinal_num` = ranking temporal medio de aparición por oportunidad
4. **Imputación**:
   - Variables numéricas sin dato = `0` (notas, importes, porcentajes)
   - `FO_rentaFam_ges__c` → media del segmento
   - `CH_MATRICULA_SUJETA_BECA` → `0` (No)
5. **Feature vinculación previa** → `max(CH_ALUMNO__PC, CH_ESTUDIANTE__PC, ...)`
6. **Drop columnas identificativas** → `ACCOUNTID, ID, ID18__PC, BIRTHDATE, CreatedDate`
   (nota: `CreatedDate` se preserva en `df_ids` como `FECHA_INICIO_ETAPA` antes de descartarse de features)
7. **Selección safe_cols** → excluye `object`, constantes, columnas PCA y `vars_excluir`

### Variables excluidas del modelo

```python
vars_excluir = [
    'desmatriculado', 'MINIMUMPAYMENTPAYED', 'CH_PAGO_SUPERIOR',
    'PL_Etapa__c', 'PL_Subetapa__c', 'ACC_DTT_FECHAULTIMAACTIVIDAD',
    'NAMEX', 'YEARPERSONBIRTHDATE', 'PAID_AMOUNT',
    'PC1', 'PC2', 'CreatedDate', 'cluster', 'interpretacion_cluster'
]
```

### Nota sobre `target` en producción

En producción el campo `target` puede no existir en `DATASET_LIMPIO`.
El preprocesador lo excluye de features automáticamente si existe,
y lo conserva en `df_ids` para permitir evaluación posterior.

---

## Predictor (src/predictor.py)

### Flujo de ejecución

```
load_dataset_limpio()  ← DATASET_LIMPIO (Oracle)
    ↓
preprocess(df, 'grado')   preprocess(df, 'master')
    ↓                              ↓
load_model(grado)          load_model(master)
    ↓                              ↓
predict_model()            predict_model()
    ↓                              ↓
prob_matricula_real        prob_matricula_real
confianza_modelo           confianza_modelo
explicacion_shap           explicacion_shap
    ↓                              ↓
         upsert_records → PMAT_PREDICTION (Oracle)
         (MERGE INTO por OPP_ID_ETAPA_COMP)
```

### Columnas calculadas

| Campo intermedio | Fórmula |
|---|---|
| `prob_matricula_real` | `prediction_score` si `prediction_label==1`, else `1 - prediction_score` |
| `confianza_modelo` | `abs(prob_matricula_real - 0.5) * 2` (0=indeciso, 1=certeza) |

| Campo en PMAT_PREDICTION | Fórmula final |
|---|---|
| `PROBABILIDAD` | `prob_matricula_real × 100` (rango 0–100, dos decimales) |
| `CONFIANZA` | `confianza_modelo × 100` (rango 0–100, dos decimales) |
| `FECHA_INICIO_ETAPA` | `CreatedDate` de `STAGE_HISTORY` en Salesforce (datetime Python → TIMESTAMP Oracle) |

---

## Tabla Oracle: PMAT_PREDICTION

```sql
CREATE TABLE PMAT_PREDICTION (
    OPP_ID_ETAPA_COMP  NVARCHAR2(200),  -- PK: OPP_ID__ETAPA__SUBETAPA
    OPP_ID             NVARCHAR2(50),   -- ID oportunidad Salesforce
    ETAPA              NVARCHAR2(100),  -- Etapa del proceso de admisión
    SUBETAPA           NVARCHAR2(100),  -- Subetapa del proceso
    FECHA_INICIO_ETAPA TIMESTAMP,       -- Fecha de entrada en la etapa (de SF STAGE_HISTORY)
    TARGET_PRED        NUMBER(1),       -- Predicción: 1=matrícula, 0=no
    TARGET_REAL        NUMBER(1),       -- Resultado real (se rellena al cierre del curso)
    PROBABILIDAD       FLOAT,           -- Probabilidad de matrícula [0–100]
    CONFIANZA          FLOAT,           -- Seguridad del modelo [0–100]
    MODELO             NVARCHAR2(20),   -- 'grado_v1' o 'master_v1'
    EXPLICACION        CLOB,            -- JSON con top-3 variables SHAP (impacto y dirección)
    FECHA_PRED         TIMESTAMP,       -- Momento de la primera predicción
    FECHA_ACTUALIZACION TIMESTAMP       -- Última actualización del registro
)
```

La tabla se crea automáticamente si no existe. Se usa **MERGE INTO** (UPSERT) con clave
`OPP_ID_ETAPA_COMP`. Solo se escribe en disco cuando cambia `PROBABILIDAD` — sin actualizaciones redundantes.

> Si la tabla ya existía sin `FECHA_INICIO_ETAPA`, el pipeline la añade automáticamente
> con `ALTER TABLE ... ADD (FECHA_INICIO_ETAPA TIMESTAMP)` la primera vez que se ejecuta.

## Vista Oracle: PMAT_PRED_ACTUAL

La fase3 crea o reemplaza automáticamente la vista tras cada UPSERT:

```sql
CREATE OR REPLACE VIEW PMATOWNER.PMAT_PRED_ACTUAL AS
SELECT p.OPP_ID, p.PROBABILIDAD, p.CONFIANZA,
       p.ETAPA, p.SUBETAPA, p.FECHA_INICIO_ETAPA, p.FECHA_ACTUALIZACION
FROM PMATOWNER.PMAT_PREDICTION p
WHERE p.FECHA_INICIO_ETAPA = (
    SELECT MAX(p2.FECHA_INICIO_ETAPA)
    FROM PMATOWNER.PMAT_PREDICTION p2
    WHERE p2.OPP_ID = p.OPP_ID
);
```

**Una fila por oportunidad**, siempre con la etapa y probabilidad más recientes.
Esta vista es la fuente de la fase4 (write-back a Salesforce).

---

## Ejecución

```bash
# Ejecutar predicciones completas (Grado + Máster)
python src/predictor.py

# Tests
pytest src/test_phase4_models_load.py -v
pytest src/test_phase4_preprocessor.py -v
pytest src/test_phase4_predictor.py -v
```

---

## Integración en pipeline.py

Esta fase corresponde a la **Fase 3** del pipeline principal:

```
pipeline.py
├── fase1 · Ingesta SF → 10 tablas Oracle (UPSERT)
├── fase2 · Limpieza → DATASET_LIMPIO (truncate + insert)
├── fase3 · Predicciones + SHAP → PMAT_PREDICTION (UPSERT) + vista PMAT_PRED_ACTUAL  ← este módulo
└── fase4 · Write-back → Salesforce NU_Probabilidad_de_matricula__c (vía sf_writer.py)
```

---

*Autor: Viewnext (Juan Velázquez y Mario Almendros)*
