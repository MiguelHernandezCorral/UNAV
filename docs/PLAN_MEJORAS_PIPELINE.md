# Plan de mejoras del pipeline — Sprint siguiente

**Autor:** Viewnext (Juan Velázquez y Mario Almendros)
**Fecha:** Marzo 2026

---

## Estado de conectividad

> ✅ **Acceso a Salesforce desbloqueado** (26-mar-2026): el puerto TCP 443 desde `hydra4-pre.unav.es` hacia `unav--fulladm.sandbox.my.salesforce.com` está operativo. El pipeline completo (fase1+fase2+fase3) puede ejecutarse desde la MV.

---

## Resumen de mejoras

| ID | Mejora | Prioridad | Dependencias | Bloqueante externo |
|---|---|---|---|---|
| M1 | Renombrar fase4 → fase3 en código y docs ✅ COMPLETADO | Alta | — | No |
| M2 | Añadir `FECHA_INICIO_ETAPA` a `PMAT_PREDICTION` ✅ COMPLETADO | Alta | M1 | No |
| M5 | Programación automática diaria a las 03:00 en la MV ✅ COMPLETADO | Alta | — | No |
| M6 | Escalar PROBABILIDAD y CONFIANZA a rango 0–100 ✅ COMPLETADO | Alta | — | No |
| M3 | Vista `PMAT_PRED_ACTUAL` (última predicción por oportunidad) ✅ COMPLETADO | Alta | M2, M6 | No |
| M4 | Write-back a Salesforce vía PATCH composite/sobjects | Alta | M3 | No — spec recibida |

**Orden de implementación recomendado:** M1 → M2 → M5 → M6 → M3 → M4

---

## M1 — Renombrar fase4 → fase3 ✅ COMPLETADO

### Motivación
Se eliminó la fase de validación (`validate`). La numeración saltaba de fase2 a fase4. Renombrar a fase3 hace la secuencia coherente: fase1 → fase2 → fase3.

### Archivos modificados
`src/pipeline.py`, `src/predictor.py`, `linux_deploy/src/pipeline.py`, `linux_deploy/src/predictor.py`, `docs/FASE3_PREDICCIONES.md` (renombrado), `docs/PIPELINE_DEV.md`, `docs/PIPELINE_CLIENTE.md`

### Impacto en producción
- Scripts o cron que usaran `--phases fase4` deben actualizarse a `--phases fase3`

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

## M5 — Programación automática diaria a las 03:00 ✅ COMPLETADO

### Motivación
El pipeline debe ejecutarse sin intervención manual cada noche para que las predicciones en Oracle estén actualizadas al inicio de la jornada laboral.

### Implementación

Se ha creado el script `linux_deploy/setup_cron.sh` que gestiona la entrada en el crontab del usuario.

**Instalar en la MV (una sola vez):**
```bash
cd /home/infra/jvelazquezc/UNAV
chmod +x setup_cron.sh
bash setup_cron.sh
```

Esto añade la línea:
```cron
0 3 * * * /home/infra/jvelazquezc/UNAV/run_pipeline.sh >> /home/infra/jvelazquezc/UNAV/logs/cron.log 2>&1
```

**Gestión del cron:**
```bash
bash setup_cron.sh --status   # ver si está activo
bash setup_cron.sh --remove   # eliminar
tail -f logs/cron.log         # seguir el log en tiempo real
```

### Archivos añadidos
- `linux_deploy/setup_cron.sh` — script de instalación/gestión del cron

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

## M6 — Escalar PROBABILIDAD y CONFIANZA a rango 0–100

### Motivación
Los modelos devuelven probabilidades en rango [0, 1]. Salesforce y los equipos de negocio trabajan con porcentajes [0, 100]. Para alinear con el campo `NU_Probabilidad_de_matricula__c` de SF y facilitar la lectura directa de la tabla Oracle, se multiplica por 100 en el momento de construir el resultado.

### Cambios en código

**`src/predictor.py` — `construir_resultado_v2()`:**
- `PROBABILIDAD`: `preds["prob_matricula_real"] * 100`
- `CONFIANZA`:    `preds["confianza_modelo"] * 100`

### Impacto

| Tabla / Vista | Antes | Después |
|---|---|---|
| `PMAT_PREDICTION.PROBABILIDAD` | 0.401 | 40.1 |
| `PMAT_PREDICTION.CONFIANZA` | 0.723 | 72.3 |
| `PMAT_PRED_ACTUAL.PROBABILIDAD` | (heredado) | 40.1 |
| SF `NU_Probabilidad_de_matricula__c` | — | 40 (entero redondeado) |

> ⚠️ La condición de cambio en el UPSERT (`compare_cols=["PROBABILIDAD"]`) sigue funcionando igual — detecta cambios en el valor ya escalado.

---

## M4 — Write-back a Salesforce vía PATCH composite/sobjects

### Motivación
Cerrar el ciclo: el pipeline devuelve las predicciones a Salesforce para que los equipos comerciales las vean directamente en el CRM sin necesidad de consultar Oracle.

### Spec del cliente (confirmada)

**Método:** `PATCH`
**Endpoint:** `{SF_URL}/services/data/v{SF_API_VERSION}/composite/sobjects`

**Cuerpo JSON:**
```json
{
  "allOrNone": false,
  "records": [
    {
      "attributes": {"type": "Opportunity"},
      "id": "0066900001j7KF0AAM",
      "NU_Probabilidad_de_matricula__c": 99
    },
    {
      "attributes": {"type": "Opportunity"},
      "id": "006Tr00000OWP7IAP",
      "NU_Probabilidad_de_matricula__c": 25
    }
  ]
}
```

- **`allOrNone: false`** — si falla un registro, los demás se procesan igualmente.
- **Máximo 200 registros por llamada** — el módulo divide en lotes de 200.
- **Campo objetivo:** `NU_Probabilidad_de_matricula__c` (valor 0–100, entero).
- **Campo de confianza:** pendiente confirmar nombre con el cliente.

### Diseño

**Nueva fase del pipeline: `fase4`** (después de fase3/predicciones).

**Nuevo módulo: `src/sf_writer.py`**

```
PMAT_PRED_ACTUAL (Oracle)
        │
        │  Filtra oportunidades donde PROBABILIDAD cambió
        │  respecto al último envío registrado en PMAT_SF_SYNC_LOG
        ▼
sf_writer.py
        │  Lotes de 200 registros → PATCH composite/sobjects
        │  Reutiliza OAuth token de sf_extractor.py
        ▼
Salesforce: actualiza NU_Probabilidad_de_matricula__c
        │
        ▼
PMAT_SF_SYNC_LOG (Oracle) — registra resultado por oportunidad
```

**Tabla de control: `PMAT_SF_SYNC_LOG`**
```sql
CREATE TABLE PMATOWNER.PMAT_SF_SYNC_LOG (
    OPP_ID           NVARCHAR2(50),
    PROBABILIDAD_ENV  FLOAT,          -- valor enviado (0-100)
    CONFIANZA_ENV     FLOAT,          -- valor enviado (0-100)
    FECHA_ENVIO       TIMESTAMP,
    STATUS            NVARCHAR2(10),  -- 'OK' / 'ERROR'
    DETALLE           NVARCHAR2(500)  -- mensaje de error si aplica
);
```

### Condición de envío
Se envía a SF **siempre que `PROBABILIDAD` haya cambiado** respecto al último valor enviado registrado en `PMAT_SF_SYNC_LOG`, independientemente de la magnitud del cambio.

### Pendiente del cliente
- [ ] Nombre del campo custom para CONFIANZA en Opportunity (si procede)
- [ ] Confirmar que el Connected App tiene permisos de escritura sobre Opportunity

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
