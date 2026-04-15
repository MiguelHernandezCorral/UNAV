# Arquitectura del pipeline UNAV

## Visión general

El pipeline es un sistema ETL + ML que corre diariamente en una MV Linux. Extrae datos de Salesforce, los limpia, aplica modelos de predicción y devuelve las probabilidades de matrícula a Salesforce.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SALESFORCE (CRM)                             │
│   Opportunity · Account · ECBS · SOLBAN · Cases · Emails · Pagos   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  OAuth2 REST API (SOQL)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FASE 1 — sf_extract_all.py                                         │
│  Extrae 10 entidades SF y hace MERGE INTO en Oracle                 │
│  Tablas: OPPORTUNITY, ACCOUNT, ECBS, SOLBAN, CASES,                │
│          EMAIL_RESULTS, ACTIVITY_HISTORY, OPP_FIELD_HISTORY,       │
│          PAGOS, STAGE_HISTORY                                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FASE 2 — cleaner.py                                                │
│  Lee tablas Oracle → limpia → construye DATASET_LIMPIO              │
│  · Renombrado de columnas                                           │
│  · Eliminación de columnas >90% NA                                  │
│  · Cálculo de target (matrícula formalizada / desmatriculado)       │
│  · Joins Opportunity × Account × ECB                                │
│  · Variables derivadas (titulación, centro, tiempos por etapa)      │
│  · Control de leakage temporal                                      │
│  Tabla resultado: DATASET_LIMPIO                                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FASE 3 — predictor.py + preprocessor.py + explainer.py            │
│  Lee DATASET_LIMPIO → predice → escribe PMAT_PREDICTION             │
│  · Segmenta por tipo: grado / master                                │
│  · Preprocesa features (preprocessor.py)                            │
│  · Carga modelos PyCaret (.pkl) desde models/                       │
│  · Calcula PROBABILIDAD (0-100) y CONFIANZA (0-100)                 │
│  · Genera explicaciones SHAP top-3 features (explainer.py)          │
│  · UPSERT por clave (OPP_ID, ETAPA, SUBETAPA)                       │
│  Tabla resultado: PMAT_PREDICTION                                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FASE 4 — sf_writer.py                                              │
│  Lee PMAT_PRED_ACTUAL → detecta cambios → PATCH a Salesforce        │
│  · Solo envía oportunidades donde cambió la probabilidad             │
│  · Escribe 2 campos en Opportunity:                                  │
│    - NU_Probabilidad_de_matricula__c (0-100, entero)                │
│    - ProbabilityConfidence__c (0-100, entero)                       │
│  · Registra cada envío en PMAT_SF_SYNC_LOG                          │
│  · Batch de 100 registros con pausa 0.5s (límites API SF)           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  PATCH composite/sobjects
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        SALESFORCE (CRM)                             │
│   Opportunity.NU_Probabilidad_de_matricula__c actualizado           │
│   Opportunity.ProbabilityConfidence__c actualizado                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tablas Oracle (esquema PMATOWNER)

| Tabla | Fase | Descripción |
|---|---|---|
| `OPPORTUNITY` | fase1 | Oportunidades de SF (datos brutos) |
| `ACCOUNT` | fase1 | Cuentas/personas de SF |
| `ECBS` | fase1 | Estudio Coste Becas |
| `SOLBAN` | fase1 | Solicitudes BASF |
| `CASES` | fase1 | Casos de SF |
| `EMAIL_RESULTS` | fase1 | Resultados de email SF |
| `ACTIVITY_HISTORY` | fase1 | Historial actividades |
| `OPP_FIELD_HISTORY` | fase1 | Historial cambios campos Opportunity |
| `PAGOS` | fase1 | Pagos registrados |
| `STAGE_HISTORY` | fase1 | Historial de etapas |
| `DATASET_LIMPIO` | fase2 | Dataset procesado listo para modelado |
| `PMAT_PREDICTION` | fase3 | Última predicción por (OPP_ID, ETAPA, SUBETAPA) |
| `PMAT_PREDICTION_HIST` | fase3 | Historial completo de predicciones (opcional) |
| `PMAT_PRED_ACTUAL` | fase3 | Vista: última predicción por OPP_ID |
| `PMAT_SF_SYNC_LOG` | fase4 | Log de envíos a Salesforce |

---

## Modelos ML

Los modelos están entrenados con PyCaret sobre el histórico de admisiones de UNAV:

| Fichero | Tipo | Descripción |
|---|---|---|
| `models/modelo_grado.pkl` | PyCaret pipeline | Predicción matrícula Grado |
| `models/modelo_master.pkl` | PyCaret pipeline | Predicción matrícula Máster |

Los modelos **no están en el repositorio** por su tamaño. Se transfieren manualmente a la MV.

---

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.9 (producción) / 3.13 (desarrollo local) |
| ML | PyCaret + CatBoost / LightGBM |
| Explicabilidad | SHAP |
| Base de datos | Oracle (oracledb thin mode) |
| CRM | Salesforce REST API v60.0 + OAuth2 |
| Scheduler | cron (06:00 diario en producción) |
| Infraestructura | MV Linux (hydra4-pre para pre, producción separada) |

---

## Flujo de credenciales

Todas las credenciales se cargan desde `.env` al arrancar cada módulo vía `python-dotenv`. Nunca se hardcodean en el código.

```
.env
├── ORA_HOST, ORA_PORT, ORA_SERVICE, ORA_USER, ORA_SCHEMA, ORA_PASSWORD
└── SF_URL, SF_CLIENT_ID, SF_CLIENT_SECRET, SF_API_VERSION, SF_PROB_FIELD, SF_CONF_FIELD
```

---

## Ejecución

El pipeline se orquesta desde `pipeline.py` que importa dinámicamente cada módulo:

```
pipeline.py
├── fase1 → sf_extract_all.run()
├── fase2 → cleaner.run()
├── fase3 → predictor.run_predictions_v2()
└── fase4 → sf_writer.run()
```

El cron ejecuta `run_pipeline.sh` que activa el venv y lanza `pipeline.py`.
