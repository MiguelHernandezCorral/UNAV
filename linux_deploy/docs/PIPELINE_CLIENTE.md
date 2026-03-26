# Pipeline de Predicción de Matrícula — Documentación para Cliente

**Proyecto:** Predicción de matrícula UNAV 2026/27
**Autor:** Viewnext (Juan Velázquez y Mario Almendros)
**Fecha:** Marzo 2026

---

## ¿Qué hace el sistema?

El pipeline extrae automáticamente los datos de candidatos desde **Salesforce**, los procesa y aplica modelos de inteligencia artificial para estimar la **probabilidad de que cada candidato finalice su matrícula**.

El resultado queda disponible en Oracle en la tabla `PMAT_PREDICTION`, lista para ser consumida desde cualquier herramienta de reporting (Power BI, Tableau, consultas SQL, etc.).

---

## Flujo del proceso

```
Salesforce CRM
      │
      │  (extracción automática vía API)
      ▼
Tablas intermedias en Oracle
      │
      │  (limpieza, enriquecimiento y anti-leakage)
      ▼
DATASET_LIMPIO (Oracle)
      │
      │  (modelos LightGBM entrenados con datos históricos)
      ▼
PMAT_PREDICTION (Oracle)   ←── resultado final
```

Cada ejecución completa tarda aproximadamente **4 minutos**.

---

## Las tres fases

### Fase 1 — Extracción de Salesforce

Se extraen **10 entidades** de Salesforce (oportunidades, cuentas, pagos, historial de etapas, emails, actividades, etc.) y se almacenan en Oracle mediante **sincronización inteligente**: solo se escriben los registros nuevos o modificados.

- 235.000+ registros sincronizados por ejecución
- Modo incremental: sin borrar ni recrear tablas

### Fase 2 — Limpieza y construcción del dataset

Se aplican más de 15 transformaciones sobre los datos brutos:

- Unión de entidades (oportunidad + cuenta + historial de etapas + pagos + actividades)
- Creación de variables derivadas (tiempo en cada etapa, porcentaje pagado, vinculación previa con la universidad, etc.)
- **Control de leakage temporal**: las variables de pago o resultado académico solo se incluyen si ya existían en el momento de cada registro histórico — esto garantiza que el modelo no "hace trampa" usando información del futuro
- Resultado: **53.357 registros × 55 columnas** listos para el modelo

### Fase 3 — Predicciones con inteligencia artificial

Se aplican dos modelos **LightGBM** (uno para Grado, otro para Máster) entrenados con datos históricos de matrículas anteriores.

Para cada candidato y etapa del proceso se genera:

| Campo | Descripción |
|---|---|
| **Probabilidad** | Entre 0 y 1 — cuán probable es que el candidato matricule |
| **Predicción** | 1 = se prevé matrícula, 0 = no se prevé |
| **Confianza** | Seguridad del modelo (0 = muy indeciso, 1 = muy seguro) |
| **Explicación** | Las 3 variables que más han influido en la predicción, con su impacto y dirección (positiva/negativa) |

**Resultados actuales (26 marzo 2026):**
- Grado: 52.305 candidatos → 20.962 con matrícula prevista (40,1 %)
- Máster: 1.052 candidatos → 807 con matrícula prevista (76,7 %)

---

## Tabla de resultados: PMAT_PREDICTION

Tabla Oracle en el esquema `PMATOWNER`. Contiene una fila por cada combinación de candidato + etapa del proceso.

| Columna | Descripción |
|---|---|
| `OPP_ID` | Identificador de la oportunidad en Salesforce |
| `ETAPA` / `SUBETAPA` | Etapa del proceso de admisión en el momento de la predicción |
| `PROBABILIDAD` | Probabilidad de matrícula [0–1] |
| `TARGET_PRED` | Predicción binaria: 1 = matrícula prevista, 0 = no |
| `TARGET_REAL` | Resultado real (se rellena al cierre del curso para evaluación del modelo) |
| `CONFIANZA` | Seguridad del modelo [0–1] |
| `EXPLICACION` | JSON con las 3 variables más influyentes, su valor de impacto y si es positivo o negativo |
| `MODELO` | Versión del modelo utilizado (`grado_v1` / `master_v1`) |
| `FECHA_PRED` | Fecha de la primera predicción |
| `FECHA_ACTUALIZACION` | Última vez que se actualizó el registro |

### Ejemplo de campo EXPLICACION

```json
[
  {"variable": "tiempo_etapa_dias",   "impacto": 0.38, "direccion": "positivo"},
  {"variable": "NU_NOTA_MEDIA_ADMISION", "impacto": 0.21, "direccion": "positivo"},
  {"variable": "PAID_PERCENT",        "impacto": -0.14, "direccion": "negativo"}
]
```

---

## Rendimiento de los modelos

Los modelos fueron validados con datos históricos de cursos anteriores:

| Métrica | Grado | Máster |
|---|---|---|
| AUC-ROC | 0,917 | 0,897 |
| Precisión global | 83,3% | 88,0% |
| Recall (detecta matrículas) | 86,1% | 95,3% |

---

## Ejecución y programación

El pipeline está desplegado en el servidor `hydra4-pre.unav.es` y puede ejecutarse:

- **Manualmente:** `bash run_pipeline.sh`
- **Programado:** vía cron o systemd (recomendado diario a las 06:00)

Cada ejecución actualiza `PMAT_PREDICTION` de forma incremental: solo modifica los registros cuya probabilidad ha cambiado respecto a la ejecución anterior.

---

## Estado del proyecto

| Fase | Estado |
|---|---|
| Extracción Salesforce → Oracle | ✅ Operativo |
| Limpieza y construcción del dataset | ✅ Operativo |
| Predicciones + explicabilidad SHAP | ✅ Operativo |
| Despliegue en servidor PRE | ✅ Operativo |
| Acceso a Salesforce desde MV (puerto 443) | ✅ Operativo |
| Programación automática diaria a las 03:00 | ✅ Configurado vía cron |
| Modelo de coste/beca | 🔜 Fase futura |
| Probabilidad de captación online | 🔜 Fase futura |

---

*Autor: Viewnext (Juan Velázquez y Mario Almendros)*
*Generado: marzo 2026*
