# Documentación del código del pipeline

## Módulos del pipeline

### `pipeline.py` — Orquestador

Punto de entrada principal. Ejecuta las fases en secuencia cargando cada módulo dinámicamente.

**Función principal:** `run_pipeline(phases, dry_run, stop_on_error, save_hist)`

**CLI:**
```bash
python src/pipeline.py                          # todas las fases
python src/pipeline.py --phases fase1 fase2     # fases concretas
python src/pipeline.py --dry-run                # sin escribir datos
python src/pipeline.py --no-stop-on-error       # continúa aunque falle una fase
python src/pipeline.py --phases fase3 --save-hist  # guarda historial
```

**PHASE_REGISTRY** — define las 4 fases:
| Fase | Módulo | Función |
|---|---|---|
| fase1 | sf_extract_all | run() |
| fase2 | cleaner | run() |
| fase3 | predictor | run_predictions_v2() |
| fase4 | sf_writer | run() |

**Logging:** fichero rotatorio en `logs/pipeline.log` (30 días retención) + consola.

---

### `sf_extract_all.py` — Fase 1: Ingesta SF → Oracle

Extrae 10 entidades de Salesforce y hace MERGE INTO en Oracle. No genera ficheros intermedios.

**Entidades extraídas:**
1. `Opportunity` → `OPPORTUNITY` — filtrada por `PL_Curso_academico__c IN ('2025/2026', '2026/2027')` y RecordType Grado/Máster
2. `Account` → `ACCOUNT`
3. `EstudioCosteBecas__c` → `ECBS`
4. `BASF_Solicitud__c` → `SOLBAN`
5. `Case` → `CASES`
6. `IndividualEmailResult__c` → `EMAIL_RESULTS`
7. `CampaignMember` → `ACTIVITY_HISTORY`
8. `OpportunityFieldHistory` → `OPP_FIELD_HISTORY`
9. `Pago__c` → `PAGOS`
10. `Historial_de_etapa__c` → `STAGE_HISTORY`

**Función principal:** `run(recreate=False)`
- `recreate=True`: elimina y recrea todas las tablas Oracle antes de insertar

**Dependencias:** `sf_extractor.py` (cliente OAuth2 SF), `oracle_connector.py`

---

### `sf_extractor.py` — Cliente Salesforce REST API

Cliente OAuth2 reutilizable para consultas SOQL a Salesforce.

**Clase:** `SalesforceExtractor`

**Métodos principales:**
- `authenticate()` — obtiene token OAuth2 (client_credentials)
- `query(soql)` — ejecuta SOQL con paginación automática (2000 registros/página)
- `query_to_dataframe(soql)` — devuelve `(records_raw, df)` con JSON aplanado

**Variables de entorno:** `SF_URL`, `SF_CLIENT_ID`, `SF_CLIENT_SECRET`, `SF_API_VERSION`

**SOQL_OPPORTUNITY** — consulta principal con ~100 campos de Opportunity incluyendo relaciones anidadas (Account, LK_Titulacion, etc.)

---

### `cleaner.py` — Fase 2: Limpieza → DATASET_LIMPIO

Replica la lógica del notebook `01_Limpieza.ipynb` leyendo y escribiendo en Oracle.

**Función principal:** `run(recreate=False)`

**Pasos internos:**
1. Carga `OPPORTUNITY`, `ACCOUNT`, `ECBS`, `STAGE_HISTORY` desde Oracle
2. Renombra columnas (mapping `OPP_RENAME` + `ACC_RENAME`)
3. Valida campos vs notebook (comparativa en log)
4. Elimina columnas con >90% NA
5. Crea target: `1` = Matrícula OOGG o Formalizada, `0` = Desmatriculado
6. Join Opportunity × Account
7. Variables derivadas: titulación, centro, año nacimiento, tipo solicitud
8. Normaliza plazo de admisión
9. Join con ECB + porcentaje pagado
10. Calcula tiempos en cada etapa (`calcular_tiempos_etapas`)
11. Control de leakage temporal (`limpiar_historial_por_hitos`)
12. Integra actividades acumuladas
13. Detecta y corrige leakage de pago
14. Elimina registros sin target
15. Selecciona columnas finales
16. UPSERT en `DATASET_LIMPIO`

---

### `preprocessor.py` — Preprocesado pre-PyCaret

Replica exactamente el preprocesado del notebook `03_Modelado.ipynb` para garantizar que las features de producción sean idénticas a las del entrenamiento.

**Funciones principales:**
- `preprocess(df, tipo, model_features)` → `(df_model, feature_names, df_ids)`
  - `tipo`: `"grado"` o `"master"`
  - Aplica etapa_compuesta, imputación lógica, feature engineering
  - Devuelve solo las features que espera el modelo
- `preparar_features_modelo(df, model_features)` → DataFrame con exactamente las columnas del modelo, en orden. Añade `0.0` para features ausentes. Deduplica columnas.

---

### `predictor.py` — Fase 3: Predicciones → PMAT_PREDICTION

**Función principal:** `run_predictions_v2(save_to_oracle, return_df, dry_run, save_hist)`

**Flujo por segmento (grado + master):**
1. Carga `DATASET_LIMPIO` desde Oracle
2. Filtra por `TIPO_SOLICITUD` (`grado` / `master`)
3. Preprocesa con `preprocessor.preprocess()`
4. Carga modelo PyCaret desde `models/modelo_{tipo}.pkl`
5. Ejecuta `predict_model()` → obtiene `prediction_label` y `prediction_score`
6. Calcula `PROBABILIDAD` = score escalado a 0-100 (entero)
7. Calcula `CONFIANZA` = distancia al umbral 0.5, escalada a 0-100
8. Genera explicaciones SHAP top-3 via `explainer.py` → JSON en columna `EXPLICACION`
9. Construye tabla con columnas: `OPP_ID_ETAPA_COMP`, `OPP_ID`, `ETAPA`, `SUBETAPA`, `TARGET_REAL`, `TARGET_PRED`, `PROBABILIDAD`, `CONFIANZA`, `MODELO`, `EXPLICACION`, `FECHA_PRED`, `FECHA_ACTUALIZACION`
10. UPSERT en `PMAT_PREDICTION` por clave `(OPP_ID, ETAPA, SUBETAPA)`
11. Crea/actualiza vista `PMAT_PRED_ACTUAL` (última predicción por OPP_ID)

---

### `explainer.py` — Explicabilidad SHAP

Genera explicaciones por fila usando SHAP TreeExplainer.

**Función principal:** `explain_row(model, row_df)` → JSON string con top-3 features

Formato del JSON:
```json
[
  {"feature": "NOTA_MEDIA", "impact": 0.34, "value": 8.5},
  {"feature": "ETAPA", "impact": -0.21, "value": "Pruebas de admisión"},
  {"feature": "CH_NACIONAL", "impact": 0.15, "value": true}
]
```

---

### `sf_writer.py` — Fase 4: Write-back → Salesforce

**Función principal:** `run(dry_run=False)`

**Flujo:**
1. Lee `PMAT_PRED_ACTUAL` (vista con última predicción por OPP_ID)
2. Lee `PMAT_SF_SYNC_LOG` para saber el último valor enviado por oportunidad
3. Filtra solo oportunidades donde cambió la PROBABILIDAD (evita envíos innecesarios)
4. Agrupa en lotes de 100 y hace PATCH a `composite/sobjects/Opportunity`
5. Escribe en cada Opportunity: `NU_Probabilidad_de_matricula__c` y `ProbabilityConfidence__c`
6. Registra resultado (OK/ERROR) en `PMAT_SF_SYNC_LOG`
7. Pausa 0.5s entre lotes para respetar límites de API de Salesforce

**Variables de entorno:** `SF_URL`, `SF_CLIENT_ID`, `SF_CLIENT_SECRET`, `SF_PROB_FIELD`, `SF_CONF_FIELD`

---

### `oracle_connector.py` — Gestión Oracle

Cliente Oracle reutilizable por todos los módulos.

**Clase:** `OracleConnector`

**Conexión:** modo proxy `ORA_USER[ORA_SCHEMA]` (thin mode, sin Oracle Client instalado)

**Métodos principales:**

| Método | Descripción |
|---|---|
| `upsert_records(records, table)` | MERGE INTO — inserta o actualiza por clave |
| `insert_records(records, table)` | INSERT histórico sin deduplicación |
| `read_table(table)` | SELECT * FROM schema.table → list[dict] |
| `count(table)` | COUNT(*) de una tabla |
| `drop_table(table)` | DROP TABLE IF EXISTS |
| `add_column_if_not_exists(table, col, type)` | ALTER TABLE ADD si no existe |
| `create_view(view, select_sql)` | CREATE OR REPLACE VIEW |

**Inferencia de tipos:** crea automáticamente las tablas Oracle inferiendo los tipos desde los valores Python (bool→NUMBER(1), int→NUMBER(10), float→FLOAT, datetime→TIMESTAMP, str→NVARCHAR2).

---

### `excel_loader.py` — Carga histórico Excel (uso puntual)

Carga el dataset histórico desde Excel a Oracle creando tablas `_HIST`. Se usa puntualmente para incorporar datos históricos al entrenamiento, no forma parte del pipeline diario.

---

## Tests

Los tests están en `src/` y cubren preprocessor, predictor, carga de modelos, sf_extractor y sf_writer. Se ejecutan con:

```bash
cd UNAV && source .venv/bin/activate
python -m pytest src/test_*.py -v
```

> Los tests no forman parte del despliegue de producción.
