# Plan de mejoras del pipeline — Sprint siguiente

**Autor:** Viewnext (Juan Velázquez y Mario Almendros)
**Fecha:** Marzo 2026

---

## Resumen de mejoras

| ID | Mejora | Prioridad | Dependencias | Bloqueante externo |
|---|---|---|---|---|
| M1 | Renombrar fase4 → fase3 en código y docs | Alta | — | No |
| M2 | Añadir `FECHA_INICIO_ETAPA` a `PMAT_PREDICTION` | Alta | M1 | No |
| M3 | Vista `PMAT_PRED_ACTUAL` (última predicción por oportunidad) | Alta | M2 | No |
| M4 | Callback a Salesforce (write-back de probabilidad y confianza) | Media | M3 | Sí — endpoint del cliente |

**Orden de implementación recomendado:** M1 → M2 → M3 → M4

---

## M1 — Renombrar fase4 → fase3

### Motivación
Se eliminó la fase de validación (`validate`). La numeración actual salta de fase2 a fase4, lo que genera confusión. Renombrar a fase3 hace la secuencia coherente.

### Archivos afectados

| Archivo | Cambio |
|---|---|
| `src/pipeline.py` | Clave `"fase4"` → `"fase3"` en PHASE_REGISTRY y PHASE_ORDER; todas las referencias internas |
| `src/predictor.py` | Docstring ("Fase 4" → "Fase 3") |
| `linux_deploy/src/pipeline.py` | Ídem (sincronizar) |
| `linux_deploy/src/predictor.py` | Ídem (sincronizar) |
| `docs/PIPELINE_DEV.md` | Tabla de fases, ejemplos de comandos |
| `docs/FASE4_PREDICCIONES.md` | Renombrar a `FASE3_PREDICCIONES.md`; actualizar referencias internas |
| `docs/PIPELINE_CLIENTE.md` | Actualizar mención de fases |
| `linux_deploy/docs/` | Sincronizar todos los docs |

### Impacto en producción
- Cambio de nombre de clave en pipeline.py: los scripts que llamen `--phases fase4` deberán actualizarse a `--phases fase3`
- Si hay cron configurado en la MV con `fase4`, actualizar la línea de cron

---

## M2 — Añadir `FECHA_INICIO_ETAPA` a `PMAT_PREDICTION`

### Motivación
Actualmente no hay forma de ordenar los registros de una misma oportunidad por antigüedad. `FECHA_INICIO_ETAPA` (= `STAGE_HISTORY.CREATEDDATE` de cada registro) permite saber cuándo entró el candidato en cada etapa, y así identificar cuál es la etapa más reciente.

### Dato de origen
`CreatedDate` ya está disponible en el pipeline: viene de `STAGE_HISTORY.CREATEDDATE`, se renombra a `CreatedDate` en `preprocessor.py` (línea 39), y se usa en `calcular_orden_automatico()` para calcular `etapa_ordinal_num`. Actualmente se descarta antes de llegar a `construir_resultado_v2`.

### Cambios en código

**`src/preprocessor.py`** — conservar `CreatedDate` en `df_ids_base`:
- En la función `preprocess()`, donde se capturan `PL_Etapa__c` y `PL_Subetapa__c` para `df_ids_base`, añadir también `CreatedDate` antes de eliminarla de las features.

**`src/predictor.py`** — añadir campo al resultado:
- En `construir_resultado_v2()`, añadir columna `FECHA_INICIO_ETAPA` tomada de `df_ids["CreatedDate"]`.

**`src/oracle_connector.py`** — no requiere cambios (inferencia de tipos automática).

**Oracle — schema PMAT_PREDICTION**:
- Primera vez: ejecutar `python src/cleaner.py --recreate` no aplica aquí.
- Usar `ALTER TABLE PMATOWNER.PMAT_PREDICTION ADD (FECHA_INICIO_ETAPA TIMESTAMP)` en Oracle antes de ejecutar el pipeline.
- O bien ejecutar el pipeline con `--recreate-pmat` si se añade ese flag (ver M2 extras).

### Resultado esperado en `PMAT_PREDICTION`

```
OPP_ID_ETAPA_COMP  | OPP_ID | ETAPA | SUBETAPA | FECHA_INICIO_ETAPA | PROBABILIDAD | ...
-------------------+--------+-------+----------+--------------------+--------------+----
ABC__Solicitud__NA | ABC    | Sol.  | NA       | 2026-01-10 08:00   | 0.32         | ...
ABC__Admitido__OK  | ABC    | Adm.  | OK       | 2026-02-15 09:30   | 0.67         | ...  ← más reciente
```

---

## M3 — Vista `PMAT_PRED_ACTUAL` (última predicción por oportunidad)

### Motivación
El cliente necesita una tabla/vista con exactamente **una fila por oportunidad** que siempre muestre la probabilidad y confianza más actualizadas (la de la etapa más reciente). Esta es la fuente para el callback a Salesforce (M4) y para reporting consolidado.

### Diseño

**Opción recomendada: Oracle VIEW** (sin coste de mantenimiento, siempre consistente).

```sql
CREATE OR REPLACE VIEW PMATOWNER.PMAT_PRED_ACTUAL AS
SELECT
    p.OPP_ID,
    p.PROBABILIDAD,
    p.CONFIANZA,
    p.ETAPA,
    p.SUBETAPA,
    p.FECHA_INICIO_ETAPA,
    p.FECHA_ACTUALIZACION
FROM PMATOWNER.PMAT_PREDICTION p
WHERE p.FECHA_INICIO_ETAPA = (
    SELECT MAX(p2.FECHA_INICIO_ETAPA)
    FROM PMATOWNER.PMAT_PREDICTION p2
    WHERE p2.OPP_ID = p.OPP_ID
);
```

- **No requiere cambios en el pipeline** — la vista se actualiza sola al hacer UPSERT en `PMAT_PREDICTION`.
- Si hay empate en `FECHA_INICIO_ETAPA` (muy raro), desempatar con `FECHA_ACTUALIZACION DESC`.

### Campos expuestos

| Campo | Descripción |
|---|---|
| `OPP_ID` | ID de la oportunidad Salesforce |
| `PROBABILIDAD` | Probabilidad de matrícula de la etapa más reciente [0–1] |
| `CONFIANZA` | Seguridad del modelo [0–1] |
| `ETAPA` / `SUBETAPA` | Etapa más reciente del candidato |
| `FECHA_INICIO_ETAPA` | Cuándo entró el candidato en esa etapa |
| `FECHA_ACTUALIZACION` | Cuándo se calculó la última predicción |

### Implementación
- Ejecutar el DDL de la vista en Oracle (una sola vez).
- Opcionalmente: añadir la creación de la vista en `oracle_connector.py` como método `create_view_if_not_exists()` para que se cree automáticamente en el primer run.

---

## M4 — Callback a Salesforce (write-back)

### Motivación
Cerrar el ciclo: el pipeline no solo lee de Salesforce, sino que le devuelve las predicciones para que los equipos comerciales las vean directamente en el CRM.

### Diseño

**Nueva fase del pipeline: `fase4`** (después de fase3/predicciones).

**Nuevo módulo: `src/sf_writer.py`**

Flujo:
```
PMAT_PRED_ACTUAL (Oracle)
        │
        │  Lee oportunidades con probabilidad cambiada
        │  (FECHA_ACTUALIZACION > última ejecución exitosa)
        ▼
sf_writer.py
        │  Para cada oportunidad: PATCH /sobjects/Opportunity/{id}
        │  Campos: {campo_prob}, {campo_confianza}
        ▼
Salesforce API (endpoint a definir por el cliente)
        │
        ▼
Log de resultados + registro de última sincronización
```

**Nueva tabla de control: `PMAT_SF_SYNC_LOG`**
```sql
CREATE TABLE PMATOWNER.PMAT_SF_SYNC_LOG (
    OPP_ID          NVARCHAR2(50),
    PROBABILIDAD_ENV FLOAT,
    CONFIANZA_ENV   FLOAT,
    FECHA_ENVIO     TIMESTAMP,
    STATUS          NVARCHAR2(10),   -- 'OK' / 'ERROR'
    DETALLE         NVARCHAR2(500)   -- mensaje de error si aplica
);
```

### Pendiente del cliente

Para implementar M4 se necesita que el cliente proporcione:

- [ ] Nombre de los **campos custom en Opportunity** donde escribir (`PMAT_Probabilidad__c`, `PMAT_Confianza__c` o similar)
- [ ] Si el Connected App ya tiene permisos de **escritura** sobre Opportunity, o hay que ampliarlos
- [ ] Confirmación de si queremos escribir cuando **cualquier campo cambia** o solo cuando `PROBABILIDAD` supera un umbral de variación (p.ej. ±0,05)

### Condición de envío recomendada
Solo se llama a SF si `|PROBABILIDAD_actual - PROBABILIDAD_env_anterior| > 0.01` — evita llamadas innecesarias por pequeñas fluctuaciones del modelo.

---

## Regla de mantenimiento continuo

Para cada cambio en el código:

1. Modificar `src/` correspondiente
2. Sincronizar `linux_deploy/src/` (copiar el fichero)
3. Actualizar la documentación técnica afectada en `docs/`
4. Sincronizar `linux_deploy/docs/` (copiar los .md)
5. Hacer commit en `ramaJuan`

---

*Autor: Viewnext (Juan Velázquez y Mario Almendros)*
