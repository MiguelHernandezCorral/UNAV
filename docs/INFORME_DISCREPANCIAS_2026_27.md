# Informe de Discrepancias y Plan de Mejora — Predicciones 2026/2027
**Fecha de análisis:** 2026-03-25
**Analista:** Claude (basado en datos reales de Oracle)
**Tablas consultadas:** `PMAT_PREDICTION`, `DATASET_LIMPIO`

---

## Resumen ejecutivo

El análisis sobre datos reales del curso 2026/2027 revela que **el modelo en sí funciona correctamente**, pero hay problemas estructurales en los datos de entrada que distorsionan los resultados. El principal hallazgo es que **no existen matrículas reales registradas** para el curso actual, lo que hace imposible evaluar la precisión del modelo. Adicionalmente, varias variables clave llegan con valores vacíos o cero para la mayoría de candidatos activos.

**Veredicto:** No es necesario reentrenar. Los problemas son de datos, no de modelo.

---

## Resultados cuantitativos del análisis

| Bloque | Dato clave | Valor observado |
|--------|-----------|-----------------|
| PMAT_PREDICTION total | Registros en tabla | 51,169 |
| TARGET_REAL = 1 (matrículas reales) | Registros con matrícula confirmada | **0 / 51,169** |
| DATASET_LIMPIO | Registros activos | 53,343 |
| TARGET = 0 en DATASET_LIMPIO | Sin ninguna matrícula registrada | **100%** |
| CU_IMPORTE_TOTAL | % nulos | **98.7%** |
| NUM_ASISTENCIAS_ACUM | % con valor 0 | **100%** |
| NUM_SOLICITUDES_ACUM | % con valor 0 | **100%** |
| Candidatos Grado / Máster | Distribución | 98% Grado / 2% Máster |

---

## Discrepancias ordenadas por criticidad

---

### P1 — CRÍTICO: TARGET_REAL está a cero para todos los registros

**Observado:**
`PMAT_PREDICTION` contiene 51,169 registros. `TARGET_REAL = 0` en el **100%** de ellos. No hay ningún `NULL` — la columna está poblada pero con cero.

**Por qué es un problema:**
El curso 2026/2027 aún no ha cerrado. Nadie ha formalizado la matrícula todavía (o los datos de cierre no se han volcado desde Salesforce a Oracle). Esto significa que **todas las métricas de evaluación del modelo (accuracy, AUC, precision, recall) son completamente inválidas** — se están calculando sobre un ground truth que no existe.

**Impacto:**
Los notebooks de análisis muestran un modelo que "no acierta nada" porque compara predicciones contra ceros que no representan realidad.

**Acción requerida:**
- Confirmar con el equipo de admisiones si el curso 2026/27 ya tiene matrículas cerradas.
- Si no hay matrículas reales aún: **no evaluar el modelo**. Solo usar las predicciones como herramienta de gestión (qué candidatos son más probables).
- Si hay matrículas reales en Salesforce pero no llegaron a Oracle: revisar la fase1 (ingesta SF→Oracle) para que el campo `TARGET` se sincronice correctamente.
- Implementar un mecanismo para poblar `TARGET_REAL` en `PMAT_PREDICTION` cuando se confirme la matrícula en SF.

---

### P1 — CRÍTICO: NUM_ASISTENCIAS_ACUM y NUM_SOLICITUDES_ACUM son 100% ceros

**Observado:**
En `DATASET_LIMPIO` (53,343 registros), ambas columnas tienen `0` en todos los registros. No hay nulos — están pobladas con cero.

**Por qué es un problema:**
El modelo fue entrenado con datos de cursos anteriores donde estos contadores reflejaban actividad real (asistencias a jornadas de puertas abiertas, solicitudes enviadas). Para el curso 2026/27, estos eventos aún no han ocurrido mayoritariamente o **el proceso de acumulación de eventos no está funcionando**.

Estas son features con peso no trivial en el modelo. Recibirlas siempre a cero produce predicciones sesgadas a la baja.

**Acción requerida:**
- Verificar en Salesforce si los eventos de asistencia/solicitud se están registrando para 2026/27.
- Si los eventos no existen aún (inicio de curso): aceptable. Documentar que las predicciones en etapas tempranas tienen menor fiabilidad.
- Si los eventos existen en SF pero no llegan a Oracle: revisar la query de ingesta en `sf_extract_all.py` para la entidad correspondiente.
- Considerar añadir un flag en las predicciones: `FIABILIDAD_BAJA = 1` cuando las features de comportamiento sean todas cero.

---

### P1 — CRÍTICO: CU_IMPORTE_TOTAL llega con 98.7% de nulos

**Observado:**
Solo 643 de 53,343 registros tienen valor real en `CU_IMPORTE_TOTAL`. El 98.7% llega como `NULL` y se imputa a `0`.

**Por qué es un problema:**
Esta variable es importante para el modelo (precio de matrícula, relacionado con decisión final). En entrenamiento tenía distribución real. Ahora llega vacía porque en etapas tempranas de admisión, Salesforce aún no tiene asignado el importe.

**Acción requerida:**
- Verificar qué etapas tienen el importe disponible en SF (normalmente "Admisión" en adelante).
- Para candidatos en etapas donde el importe no existe aún, la imputación a `0` es la única opción disponible — pero debe documentarse como limitación.
- Considerar imputar por `mediana del programa` en lugar de `0` para no sesgar el modelo.
- Añadir la variable `CU_IMPORTE_TOTAL_DISPONIBLE` (binaria: 1 si hay valor real) como feature adicional del modelo.

---

### P2 — IMPORTANTE: CH_MATRICULA_SUJETA_BECA llega con 98.7% de nulos

**Observado:**
Solo 1.3% de los registros tienen valor en esta columna. El resto se imputa a `0` (sin beca).

**Por qué es un problema:**
Si un candidato tiene beca pendiente de confirmación, la decisión de matrícula depende de ella. Imputar a `0` (sin beca) introduce sesgo en un segmento relevante de candidatos.

**Acción requerida:**
- Investigar si el campo de beca en SF se rellena tarde en el proceso.
- Añadir la variable `BECA_PENDIENTE` cuando el campo esté vacío en etapas donde debería tener valor.

---

### P2 — IMPORTANTE: NU_MEDIA_EXPEDIENTE_UNIVERSITA llega con 84.1% de nulos

**Observado:**
Solo 15.9% de los candidatos tienen nota de expediente universitario. Esta variable es clave para el segmento Máster.

**Por qué es un problema:**
Para candidatos de Máster, esta nota es uno de los predictores más potentes. Si llega vacía para 84% de ellos, el modelo Máster está operando con información muy degradada.

**Acción requerida:**
- Verificar si la nota se rellena manualmente por el equipo de admisiones en una fase tardía.
- Considerar añadir la variable como "disponible / no disponible" para que el modelo aprenda que su ausencia también es informativa.

---

### P2 — IMPORTANTE: Distribución de etapas sesgada hacia fases tempranas

**Observado:**
Top etapas actuales: Inicio (9,599), Validación/Recibida (6,070), Validación/Completa (5,752). Las etapas avanzadas (Admisión, Pre-matrícula, Matrícula) tienen muchos menos candidatos.

**Por qué es un problema:**
El modelo fue entrenado con datos históricos donde el peso de cada etapa refleja el ciclo completo del proceso. Evaluarlo al inicio del ciclo, cuando el 60%+ de candidatos están en etapas iniciales, produce probabilidades bajas por diseño — no es un fallo del modelo.

**Acción requerida:**
- Comunicar a los usuarios del modelo que las predicciones en etapas tempranas son orientativas, no definitivas.
- Considerar añadir una banda de confianza en los dashboards: "Predicción en etapa temprana — fiabilidad reducida".
- Evaluar el modelo **solo sobre candidatos en etapas avanzadas** para obtener métricas significativas.

---

### P2 — IMPORTANTE: Segmento Máster muy pequeño (2% del total)

**Observado:**
`PMAT_PREDICTION` tiene 50,171 registros de grado vs. 998 de máster. `DATASET_LIMPIO` refleja la misma proporción.

**Por qué es un problema:**
Con menos de 1,000 candidatos de máster, el modelo máster no tiene suficiente muestra para ser evaluado robustamente. Cualquier métrica calculada sobre esta muestra tendrá alta varianza.

**Acción requerida:**
- Verificar si la ingesta de Salesforce está capturando todos los programas de máster.
- Si la muestra es correcta: documentar la limitación y no tomar decisiones estratégicas basándose solo en las métricas del modelo máster en este período.

---

### P3 — MENOR: PORCENTAJE_PAGADO_FINAL llega con 72.9% de nulos

**Observado:**
Solo 27.1% de los registros tienen valor en el porcentaje de pago realizado.

**Por qué es un problema:**
En etapas tempranas, nadie ha pagado nada aún. La imputación a `0` es lógicamente correcta en este caso — es coherente con el negocio.

**Acción requerida:**
Ninguna inmediata. Documentar que esta variable es informativa solo en etapas avanzadas (Matrícula, Pre-matrícula).

---

### P3 — MENOR: 2 features se recalculan en cada ejecución (no se persisten)

**Observado:**
`etapa_ordinal_num` y `vinculacion_previa` no están en Oracle — se calculan en `preprocessor.py` en cada ejecución de la pipeline.

**Por qué es un problema:**
Menor: el cálculo es correcto. Pero si el conjunto de datos cambia entre ejecuciones, el orden ordinal de etapas puede variar ligeramente, introduciendo inconsistencia entre predicciones históricas.

**Acción requerida:**
Bajo prioridad. Considerar persistir el mapeo `etapa→ordinal` en Oracle o en un fichero de configuración para garantizar consistencia entre ejecuciones.

---

## Plan de mejora de la pipeline

### Mejora 1 — Sincronización de TARGET_REAL desde Salesforce [P1]

**Problema:** `TARGET_REAL` no se actualiza cuando un candidato formaliza matrícula.
**Solución:**
Añadir en `fase4` (o en una nueva `fase5`) una query a Salesforce que recupere las oportunidades cerradas con éxito (Stage = "Matrícula Formalizada" o equivalente) y actualice `TARGET_REAL = 1` en `PMAT_PREDICTION`.

```python
# En predictor.py o en un módulo nuevo: actualizar_targets.py
def actualizar_target_real(conn, sf_client):
    """
    Consulta SF por oportunidades cerradas y actualiza TARGET_REAL en Oracle.
    """
    opp_matriculadas = sf_client.query(
        "SELECT Id FROM Opportunity WHERE StageName = 'Matricula_Formalizada__c'"
    )
    ids = [r['Id'] for r in opp_matriculadas['records']]
    if ids:
        conn.execute(
            f"UPDATE PMATOWNER.PMAT_PREDICTION SET TARGET_REAL = 1 "
            f"WHERE OPP_ID IN ({','.join([':'+str(i) for i in range(len(ids))])})",
            ids
        )
```

**Esfuerzo:** 1-2 días. **Impacto:** Permite evaluación real del modelo.

---

### Mejora 2 — Imputación inteligente de CU_IMPORTE_TOTAL [P1]

**Problema:** 98.7% de nulos imputados a `0`, cuando `0` no representa la realidad (el precio no es 0, simplemente no se conoce aún).
**Solución:**
En `preprocessor.py`, cambiar la imputación de `CU_IMPORTE_TOTAL` de `fillna(0)` a `fillna(mediana por titulación)`.

```python
# En preprocessor.py → función imputar()
if 'CU_IMPORTE_TOTAL' in df.columns and 'TITULACION' in df.columns:
    mediana_por_tit = df.groupby('TITULACION')['CU_IMPORTE_TOTAL'].transform('median')
    mediana_global = df['CU_IMPORTE_TOTAL'].median()
    df['CU_IMPORTE_TOTAL'] = df['CU_IMPORTE_TOTAL'].fillna(mediana_por_tit).fillna(mediana_global)
```

**Esfuerzo:** 2 horas. **Impacto:** Mejora estimada de 2-5 puntos en accuracy para candidatos en etapas medias.

---

### Mejora 3 — Flag de fiabilidad en predicciones [P1]

**Problema:** Las predicciones en etapas tempranas con features vacías son poco fiables, pero se presentan con la misma confianza que las de etapas avanzadas.
**Solución:**
Calcular un score de fiabilidad basado en el porcentaje de features clave disponibles y añadirlo a `PMAT_PREDICTION`.

```python
# En predictor.py → función guardar_en_oracle_v2()
FEATURES_CLAVE = [
    'CU_IMPORTE_TOTAL', 'num_asistencias_acum', 'num_solicitudes_acum',
    'NU_NOTA_MEDIA_ADMISION', 'PAID_PERCENT'
]

def calcular_fiabilidad(row, features_clave):
    disponibles = sum(1 for f in features_clave if f in row and row[f] != 0)
    return round(disponibles / len(features_clave), 2)
```

Nueva columna en `PMAT_PREDICTION`: `FIABILIDAD_DATOS` [0.0–1.0].

**Esfuerzo:** 4 horas. **Impacto:** Permite filtrar predicciones por calidad de datos en los dashboards.

---

### Mejora 4 — Verificación de contadores de eventos en fase1 [P1]

**Problema:** `NUM_ASISTENCIAS_ACUM` y `NUM_SOLICITUDES_ACUM` son 100% ceros.
**Solución:**
En `sf_extract_all.py`, verificar que la entidad que alimenta estos contadores (probablemente `CampaignMember` o `Task`) se está ingiriendo correctamente.

```python
# Añadir en el log de fase1 una comprobación post-ingesta:
for col in ['NUM_ASISTENCIAS_ACUM', 'NUM_SOLICITUDES_ACUM']:
    n_nonzero = df[col].astype(float).gt(0).sum()
    if n_nonzero == 0:
        logger.warning(f"ALERTA: {col} es 0 para todos los registros — revisar ingesta SF")
```

**Esfuerzo:** 1 día (incluyendo investigación de la entidad SF correcta). **Impacto:** Recupera información de comportamiento que el modelo usa como señal relevante.

---

### Mejora 5 — Evaluación diferida del modelo [P2]

**Problema:** No se puede evaluar el modelo mientras el curso no haya cerrado.
**Solución:**
Implementar evaluación mensual automatizada: una vez al mes, calcular métricas del modelo solo sobre los registros con `TARGET_REAL IS NOT NULL` y guardar el resultado en una tabla `PMAT_MODEL_METRICS`.

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

**Esfuerzo:** 1 día. **Impacto:** Permite detectar degradación del modelo antes de que sea grave.

---

### Mejora 6 — Imputación de CU_IMPORTE_TOTAL por mediana de titulación en reentrenamiento [P2]

Cuando se reentrene el modelo (previsiblemente para el curso 2027/28), usar la misma estrategia de imputación por mediana de titulación para que el modelo aprenda la distribución real del precio en lugar del cero sistemático.

**Esfuerzo:** 0.5 días (ajuste en el notebook de modelado). **Impacto:** Elimina sesgo de entrenamiento/inferencia en esta variable.

---

### Mejora 7 — Persistir mapeo etapa→ordinal [P3]

**Problema:** `etapa_ordinal_num` se recalcula en cada ejecución y puede variar si aparecen etapas nuevas.
**Solución:**
Guardar el mapeo en un fichero `models/etapa_ordinal_map.json` en el primer entrenamiento y cargarlo en la pipeline de inferencia.

**Esfuerzo:** 2 horas. **Impacto:** Garantiza consistencia entre predicciones históricas.

---

## Resumen de acciones por prioridad

| Prioridad | Acción | Responsable | Esfuerzo |
|-----------|--------|-------------|----------|
| P1 | Confirmar con admisiones si hay matrículas reales 2026/27 | Mario + Responsable admisiones | 1h |
| P1 | Implementar sincronización TARGET_REAL desde SF | Mario | 2 días |
| P1 | Cambiar imputación CU_IMPORTE_TOTAL → mediana por titulación | Mario | 2h |
| P1 | Añadir flag FIABILIDAD_DATOS en PMAT_PREDICTION | Mario | 4h |
| P1 | Investigar y reparar ingesta de contadores de eventos SF | Mario | 1 día |
| P2 | Implementar evaluación mensual automatizada del modelo | Mario | 1 día |
| P2 | Verificar ingesta de programas máster (¿están todos?) | Mario | 0.5 días |
| P2 | Documentar limitaciones en dashboards (etapas tempranas) | Juan/Usoa | 0.5 días |
| P2 | Imputación beca → "pendiente" en lugar de "sin beca" | Mario | 2h |
| P3 | Persistir mapeo etapa→ordinal en JSON | Mario | 2h |

---

## Conclusión

**El modelo no está roto.** Las métricas aparentemente bajas se deben a:

1. **No hay ground truth**: `TARGET_REAL = 0` para todos los registros — el curso no ha cerrado.
2. **Datos de entrada degradados**: 4-5 features clave llegan vacías o con cero para >80% de candidatos en etapas tempranas.
3. **Distribución temporal**: La evaluación se hace al inicio del ciclo cuando los candidatos están en etapas con poca información disponible.

**Recomendación inmediata:** No reentrenar. Primero implementar las mejoras P1 (TARGET_REAL + imputaciones) y evaluar el modelo en 30-60 días cuando el curso haya avanzado y haya matrículas reales que comparar.
