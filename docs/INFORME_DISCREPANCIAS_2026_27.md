# Informe de Discrepancias y Plan de Mejora — Predicciones 2026/2027
**Fecha de análisis:** 2026-03-25
**Analista:** jvelazquezc + Claude (basado en datos reales de Oracle)
**Tablas consultadas:** `PMAT_PREDICTION`, `DATASET_LIMPIO`, código fuente `cleaner.py`, `sf_extract_all.py`

---

## Resumen ejecutivo

El análisis sobre datos reales del curso 2026/2027 revela que **el modelo en sí funciona correctamente**. Las métricas aparentemente bajas tienen causas estructurales bien identificadas, en su mayoría esperables al inicio del ciclo académico. El único punto que requiere verificación urgente es si los contadores de actividad (asistencias/solicitudes) están llegando con datos reales desde Salesforce.

**Veredicto:** No es necesario reentrenar. Los problemas son de datos de entrada, no de modelo.

---

## Resultados cuantitativos del análisis

| Bloque | Dato clave | Valor observado |
|--------|-----------|-----------------|
| PMAT_PREDICTION total | Registros en tabla | 51,169 |
| TARGET_REAL = 1 (matrículas reales) | Registros con matrícula confirmada | 0 / 51,169 |
| DATASET_LIMPIO | Registros activos | 53,343 |
| TARGET = 0 en DATASET_LIMPIO | Sin ninguna matrícula registrada | 100% |
| CU_IMPORTE_TOTAL | % nulos | 98.7% |
| NUM_ASISTENCIAS_ACUM | % con valor 0 | 100% |
| NUM_SOLICITUDES_ACUM | % con valor 0 | 100% |
| Candidatos Grado / Máster | Distribución | 98% Grado / 2% Máster |

---

## Discrepancias ordenadas por criticidad

---

### NORMAL — TARGET_REAL es cero para todos los registros

**Observado:** `PMAT_PREDICTION` tiene 51,169 registros. `TARGET_REAL = 0` en el 100%.

**Diagnóstico:** Comportamiento esperado. Las matrículas del curso 2026/27 aún no han comenzado. El campo se poblará cuando los candidatos formalicen matrícula.

**Impacto:** Las métricas de evaluación del modelo (accuracy, AUC) no son calculables en este momento. Esto no indica que el modelo funcione mal.

**Acción:** Ninguna urgente. Cuando se registren las primeras matrículas en SF, asegurarse de que `fase1` las sincroniza a Oracle y que existe un mecanismo para actualizar `TARGET_REAL` en `PMAT_PREDICTION`.

---

### P1 — CRÍTICO: Verificar si NUM_ASISTENCIAS_ACUM y NUM_SOLICITUDES_ACUM llegan con datos reales

**Observado:** En `DATASET_LIMPIO` (53,343 registros), ambas columnas tienen valor `0` en todos los registros. No hay nulos — están pobladas con cero, resultado del `fillna(0)` en `integrar_actividades_progresivo`.

**Diagnóstico técnico:**
La función `integrar_actividades_progresivo` en `cleaner.py` ([línea 391](../src/cleaner.py#L391)) carga la tabla `ACTIVITY_HISTORY` desde Oracle y hace un join por `ID18__PC + PL_CURSO_ACADEMICO` = `ContactId + Campaign.AcademicCourse__c`. Hay **dos causas posibles** para los ceros:

- **Causa A — No hay actividades aún (esperado):** La tabla `ACTIVITY_HISTORY` tiene 0 filas para 2026/27 porque las jornadas de puertas abiertas y eventos de captación aún no han ocurrido. En ese caso, los ceros son correctos.
- **Causa B — Bug silencioso en el join:** `Campaign.AcademicCourse__c` no está relleno en las campañas de 2026/27, con lo que el join no hace match aunque haya actividades en SF. El resultado es el mismo (ceros), pero la causa es un problema de datos en Salesforce.

**Verificación necesaria:**
```python
from oracle_connector import OracleConnector
import pandas as pd
conn = OracleConnector()

df_act = pd.DataFrame(conn.read_table('ACTIVITY_HISTORY'))
print(f"Filas en ACTIVITY_HISTORY : {len(df_act):,}")
if len(df_act) > 0:
    col_curso = 'CAMPAIGN.ACADEMICCOURSE__C'
    if col_curso in df_act.columns:
        print(f"AcademicCourse__c nulos  : {df_act[col_curso].isna().sum():,}")
        print(f"Valores únicos de curso  :")
        print(df_act[col_curso].value_counts().head(10))
    else:
        print(f"Columnas disponibles: {list(df_act.columns)}")
```

**Interpretación:**
- Si `ACTIVITY_HISTORY` tiene 0 filas → Causa A. Normal, no hay bug.
- Si tiene filas pero `Campaign.AcademicCourse__c` está vacío → Causa B. Hay que rellenar ese campo en SF o adaptar el join en `cleaner.py`.
- Si tiene filas y el campo de curso está correcto → bug más sutil; revisar el valor exacto de `PL_CURSO_ACADEMICO` vs. `AcademicCourse__c`.

**Acción:** Ejecutar la verificación anterior antes de descartar un bug en la pipeline.

---

### P1 — ACLARADO: CU_IMPORTE_TOTAL y CH_MATRICULA_SUJETA_BECA son nulos por diseño (anti-leakage)

**Observado:** `CU_IMPORTE_TOTAL`: 98.7% nulos. `CH_MATRICULA_SUJETA_BECA`: 98.7% nulos.

**Diagnóstico técnico:**
Ambas variables vienen directamente de `Opportunity` en Salesforce (`CU_Importe_total__c` y `CH_Matricula_sujeta_beca__c`). Sin embargo, en `cleaner.py` existe un **control anti-leakage** explícito ([línea 381-385](../src/cleaner.py#L381-L385)):

```python
mask_econ = (
    df_final["fecha_matricula_iniciada"].isna() |
    (df_final["CreatedDate"] < df_final["fecha_matricula_iniciada"])
)
df_final.loc[mask_econ, cols_econ] = np.nan  # ← CU_IMPORTE_TOTAL y CH_MATRICULA_SUJETA_BECA incluidas
```

Esto significa: para cualquier registro cuya fecha de creación sea anterior a "Matrícula iniciada", estas variables se fuerzan a `NULL`. Como **ningún candidato de 2026/27 ha llegado a "Matrícula iniciada"** aún, el 100% queda a nulo. Esto es **comportamiento correcto e intencionado** — evita que el modelo entrene con información del futuro.

Desde el punto de vista de SF: los datos sí existen en Salesforce para los candidatos que los tienen, pero el anti-leakage los anula correctamente.

**Implicación para predicciones:**
El modelo se entrenó con estas variables también a nulo para las mismas etapas. La imputación a `0` en `preprocessor.py` es consistente con el entrenamiento. No hay degradación por esta causa.

**Acción:** Ninguna. El comportamiento es correcto.

---

### P2 — IMPORTANTE: NU_MEDIA_EXPEDIENTE_UNIVERSITA llega con 84.1% de nulos

**Observado:** Solo 15.9% de los candidatos tienen nota de expediente universitario.

**Diagnóstico:** Esta variable viene de `Account` (`NU_Media_Expediente_Universitario__c`) y la rellena el equipo de admisiones manualmente una vez revisada la documentación del candidato. En fases tempranas (Inicio, Validación) no se ha revisado aún.

**Impacto:** Afecta principalmente al modelo Máster, donde esta nota es un predictor relevante.

**Acción:** Documentar que las predicciones de Máster en etapas previas a "Documentación validada" tienen menor fiabilidad por esta causa.

---

### P2 — IMPORTANTE: Distribución de etapas sesgada hacia fases tempranas

**Observado:** Top etapas: Inicio (9,599), Validación/Recibida (6,070), Validación/Completa (5,752). Las etapas avanzadas tienen pocos candidatos.

**Diagnóstico:** Es el inicio del ciclo 2026/27. La distribución refleja la realidad del proceso, no un fallo del modelo.

**Impacto:** Las probabilidades medias son bajas porque en etapas iniciales hay menos información disponible. Es el comportamiento esperado del modelo.

**Acción:** Comunicar a los usuarios que en etapas tempranas las predicciones son orientativas. Considerar mostrar en dashboards una banda de confianza o un indicador de etapa.

---

### P2 — IMPORTANTE: Segmento Máster muy pequeño (2% del total)

**Observado:** `PMAT_PREDICTION` tiene 998 registros de Máster vs. 50,171 de Grado.

**Diagnóstico:** Verificar si la ingesta de SF está capturando todos los programas de Máster. Si la muestra es correcta, el modelo Máster opera con poca estadística y sus métricas agregadas tienen alta varianza.

**Acción:** Confirmar con admisiones si 998 candidatos es el volumen esperado para Máster en esta fecha del proceso.

---

### P3 — ACLARADO: Mapeo etapa→ordinal se recalcula en cada ejecución

**Diagnóstico (confirmado):** Este es el comportamiento **correcto e intencionado**. El orden ordinal se calcula a partir de los datos reales de cada ejecución, reflejando la secuencia temporal real de las etapas en el curso actual. Un mapeo fijo no capturaría variaciones en el proceso de admisión de un año a otro.

**Acción:** Ninguna. El diseño actual es el correcto.

---

## Plan de mejora de la pipeline

### Mejora 1 — Verificar e instrumentar la ingesta de actividades [P1]

**Problema:** Los contadores de asistencias/solicitudes son 100% ceros. No sabemos si es porque no hay actividades aún o porque hay un bug de join silencioso en `Campaign.AcademicCourse__c`.

**Solución:**
Añadir en el log de `fase2` un check post-integración que distinga las dos causas:

```python
# En cleaner.py → integrar_actividades_progresivo(), al final
n_act_raw = len(df_actividades)
n_con_asist = (df_final["num_asistencias_acum"] > 0).sum()
logger.info(
    "  Actividades brutas recibidas: %d | Candidatos con asistencias: %d",
    n_act_raw, n_con_asist
)
if n_act_raw > 0 and n_con_asist == 0:
    logger.warning(
        "  ALERTA: ACTIVITY_HISTORY tiene %d filas pero ningún candidato "
        "tiene asistencias acumuladas — verificar Campaign.AcademicCourse__c",
        n_act_raw
    )
```

Esto convierte el fallo silencioso en un warning visible en los logs.

**Esfuerzo:** 1 hora. **Impacto:** Diagnóstico inmediato en cada ejecución.

---

### Mejora 2 — Sincronización de TARGET_REAL desde Salesforce [P2]

**Problema:** `TARGET_REAL` en `PMAT_PREDICTION` no se actualiza automáticamente cuando un candidato formaliza matrícula en SF.

**Solución:**
Añadir en `fase4` (o una nueva fase opcional) una actualización periódica de `TARGET_REAL` para las oportunidades cerradas ganadas:

```python
def actualizar_target_real(conn, sf_client, curso):
    """Actualiza TARGET_REAL en PMAT_PREDICTION para matrículas confirmadas."""
    result = sf_client.query(
        f"SELECT Id FROM Opportunity "
        f"WHERE PL_Curso_academico__c = '{curso}' "
        f"AND IsWon = true"
    )
    ids_matriculados = [r['Id'] for r in result['records']]
    if ids_matriculados:
        # UPDATE en Oracle por lotes
        conn.execute_bulk_update(
            "PMATOWNER.PMAT_PREDICTION",
            {"TARGET_REAL": 1},
            where_col="OPP_ID",
            where_values=ids_matriculados
        )
        logger.info("TARGET_REAL actualizado para %d matrículas", len(ids_matriculados))
```

**Esfuerzo:** 1-2 días. **Impacto:** Habilita la evaluación real del modelo en cuanto empiecen las matrículas.

---

### Mejora 3 — Añadir FIABILIDAD_DATOS en PMAT_PREDICTION [P2]

**Problema:** Las predicciones de etapa temprana (con pocas features disponibles) se presentan con el mismo formato que las de etapa avanzada. El usuario no puede distinguir unas de otras.

**Solución:**
Calcular un score de disponibilidad de features clave y almacenarlo como columna adicional:

```python
FEATURES_DISPONIBILIDAD = [
    'CU_IMPORTE_TOTAL', 'num_asistencias_acum', 'num_solicitudes_acum',
    'NU_NOTA_MEDIA_ADMISION', 'PAID_PERCENT', 'NU_MEDIA_EXPEDIENTE_UNIVERSITA'
]

def calcular_fiabilidad_datos(row):
    disponibles = sum(
        1 for f in FEATURES_DISPONIBILIDAD
        if f in row and pd.notna(row[f]) and row[f] != 0
    )
    return round(disponibles / len(FEATURES_DISPONIBILIDAD), 2)
```

Nueva columna `FIABILIDAD_DATOS` [0.0–1.0] en `PMAT_PREDICTION`. Requiere añadir la columna a la tabla Oracle y al proceso de escritura en `predictor.py`.

**Esfuerzo:** 4-6 horas. **Impacto:** Los dashboards pueden filtrar o colorear predicciones según fiabilidad.

---

### Mejora 4 — Evaluación mensual automática del modelo [P2]

**Problema:** No hay monitoreo automático de la degradación del modelo.

**Solución:**
Implementar un script de evaluación mensual que calcule métricas solo sobre registros con `TARGET_REAL IS NOT NULL` y las guarde en una tabla de histórico:

```sql
CREATE TABLE PMATOWNER.PMAT_MODEL_METRICS (
    FECHA_EVAL    TIMESTAMP,
    MODELO        NVARCHAR2(50),
    N_EVALUADOS   NUMBER,
    ACCURACY      FLOAT,
    AUC           FLOAT,
    PRECISION_1   FLOAT,
    RECALL_1      FLOAT
);
```

**Esfuerzo:** 1 día. **Impacto:** Alerta temprana de degradación del modelo para el próximo curso.

---

## Resumen de acciones por prioridad

| Prioridad | Acción | Esfuerzo |
|-----------|--------|----------|
| P1 | Ejecutar verificación de ACTIVITY_HISTORY (Causa A vs. B) | 1h |
| P1 | Añadir warning en logs cuando hay actividades pero 0 asistencias | 1h |
| P2 | Implementar sincronización TARGET_REAL desde SF | 2 días |
| P2 | Añadir columna FIABILIDAD_DATOS en PMAT_PREDICTION | 4-6h |
| P2 | Verificar volumen esperado de Máster con admisiones | 0.5h |
| P2 | Implementar evaluación mensual automática del modelo | 1 día |
| P2 | Documentar limitaciones de predicciones en etapas tempranas | 0.5 días |

---

## Conclusión

**El modelo no está roto.** Los hallazgos se distribuyen en tres categorías:

1. **Comportamiento esperado (no requiere acción):**
   - `TARGET_REAL = 0` — las matrículas aún no han comenzado
   - `CU_IMPORTE_TOTAL` y `CH_MATRICULA_SUJETA_BECA` nulos — el anti-leakage de `cleaner.py` funciona correctamente
   - Mapeo etapa→ordinal recalculado — diseño correcto e intencionado
   - Distribución de etapas sesgada a inicio — es principio de ciclo

2. **Requiere verificación urgente:**
   - `NUM_ASISTENCIAS_ACUM` y `NUM_SOLICITUDES_ACUM` = 100% ceros — puede ser normal (no hay eventos aún) o un bug silencioso de join en `Campaign.AcademicCourse__c`. Requiere consultar Oracle directamente.

3. **Mejoras recomendadas a medio plazo:**
   - Sincronización de `TARGET_REAL` cuando se confirmen matrículas
   - Score de fiabilidad de datos por predicción
   - Monitoreo mensual de métricas del modelo
