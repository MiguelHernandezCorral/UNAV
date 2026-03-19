# Fase 4 · Predicciones con Modelos Preentrenados

## Descripción general

Esta fase usa los modelos PyCaret preentrenados (`modelo_final_grado.pkl` y
`modelo_final_master.pkl`) para generar predicciones de probabilidad de matrícula
sobre el dataset limpio almacenado en Oracle (`DATASET_LIMPIO`).

El resultado se almacena en la tabla Oracle `PREDICCIONES`, manteniendo un histórico
completo de cada ejecución.

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
load_dataset_limpio()
    ↓
preprocess(df, 'grado')   preprocess(df, 'master')
    ↓                              ↓
load_model(grado)          load_model(master)
    ↓                              ↓
predict_model()            predict_model()
    ↓                              ↓
prob_matricula_real        prob_matricula_real
confianza_modelo           confianza_modelo
    ↓                              ↓
         insert_records → PREDICCIONES (Oracle)
```

### Columnas calculadas

| Campo | Fórmula |
|---|---|
| `prob_matricula_real` | `prediction_score` si `prediction_label==1`, else `1 - prediction_score` |
| `confianza_modelo` | `abs(prob_matricula_real - 0.5) * 2` (0=indeciso, 1=certeza) |

---

## Tabla Oracle: PREDICCIONES

```sql
CREATE TABLE PREDICCIONES (
    OPP_ID       NVARCHAR2(50),    -- ID de la oportunidad Salesforce
    PROBABILIDAD FLOAT,            -- Probabilidad de matrícula [0, 1]
    TARGET_PRED  NUMBER(1),        -- Predicción binaria: 0=No matrícula, 1=Matrícula
    CONFIANZA    FLOAT,            -- Confianza del modelo [0, 1]
    MODELO       NVARCHAR2(10),    -- 'grado' o 'master'
    FECHA_PRED   TIMESTAMP         -- Momento de la predicción
)
```

La tabla se crea automáticamente si no existe (vía `OracleConnector.create_table_if_not_exists`).
Se usa **INSERT** (no UPSERT) para mantener historial completo por ejecución.

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

Esta fase corresponde a la **Fase 4** del pipeline principal:

```
pipeline.py
├── Fase 1 · Ingesta SF
├── Fase 2 · Upsert Oracle
├── Fase 3 · Limpieza → DATASET_LIMPIO
├── Fase 4 · Predicciones ← este módulo
└── Fase 5 · (futuro) Volcado resultados adicionales
```
