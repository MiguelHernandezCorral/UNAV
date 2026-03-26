# Operacional del pipeline UNAV

**Fecha de revisión:** 26-mar-2026
**Autor:** Viewnext (Juan Velázquez y Mario Almendros)

---

## 1. Cron diario — cobertura de fases

El cron instalado con `setup_cron.sh` lanza `run_pipeline.sh` sin argumentos,
lo que equivale a ejecutar todas las fases:

```
fase1 → fase2 → fase3 → fase4
```

**fase4 (write-back a Salesforce) ya está incluida** en la ejecución nocturna.
No se requiere ningún cambio en el cron.

### Gestión del cron

```bash
bash setup_cron.sh              # instalar (una sola vez)
bash setup_cron.sh --status     # verificar que está activo
bash setup_cron.sh --remove     # eliminar si es necesario
tail -f logs/cron.log           # seguir el log en tiempo real
```

Entrada crontab instalada:
```cron
0 3 * * * /home/infra/jvelazquezc/UNAV/run_pipeline.sh >> /home/infra/jvelazquezc/UNAV/logs/cron.log 2>&1
```

### Ejecución puntual sin esperar al cron

```bash
# Desde la MV, con el venv activado
source .venv/bin/activate
bash run_pipeline.sh                           # pipeline completo
bash run_pipeline.sh --phases fase4            # solo write-back a SF
bash run_pipeline.sh --phases fase3 fase4      # predicciones + write-back
```

---

## 2. Retención de PMAT_SF_SYNC_LOG

La tabla `PMAT_SF_SYNC_LOG` crece con cada ejecución: registra un fila por
oportunidad enviada a Salesforce (OK o ERROR). Sin mantenimiento, acumulará
~10.000 filas por ejecución diaria, lo que supone ~3,6 millones de filas al año.

### Política recomendada: purga mensual de registros > 90 días

Ejecutar en Oracle (DBA o tarea programada):

```sql
-- Eliminar registros de más de 90 días de antigüedad
DELETE FROM PMATOWNER.PMAT_SF_SYNC_LOG
WHERE FECHA_ENVIO < SYSTIMESTAMP - INTERVAL '90' DAY;

COMMIT;
```

### Opciones de automatización

**Opción A — Oracle DBMS_SCHEDULER (recomendada para producción):**
```sql
BEGIN
  DBMS_SCHEDULER.CREATE_JOB(
    job_name   => 'PURGA_SF_SYNC_LOG',
    job_type   => 'PLSQL_BLOCK',
    job_action => 'BEGIN
                     DELETE FROM PMATOWNER.PMAT_SF_SYNC_LOG
                     WHERE FECHA_ENVIO < SYSTIMESTAMP - INTERVAL ''90'' DAY;
                     COMMIT;
                   END;',
    repeat_interval => 'FREQ=MONTHLY;BYDAY=1;BYHOUR=4',
    enabled    => TRUE,
    comments   => 'Purga mensual del log de sincronización SF'
  );
END;
/
```

**Opción B — Script Python periódico (más sencillo):**
```python
# Añadir a sf_writer.py o ejecutar manualmente
conn.execute_ddl(
    "DELETE FROM PMAT_SF_SYNC_LOG WHERE FECHA_ENVIO < SYSTIMESTAMP - INTERVAL '90' DAY"
)
```

### Acción recomendada
Solicitar a infraestructura Oracle que creen el job `PURGA_SF_SYNC_LOG`.
Si no es posible, programar el script Python via cron mensualmente.

---

## 3. Campo CONFIANZA en Salesforce — ✅ IMPLEMENTADO (26-mar-2026)

Confirmado por Usoa Gómez (cliente) el 26-mar-2026. El pipeline envía ahora
dos campos a Salesforce en cada write-back:

| Campo SF | Variable entorno | Valor |
|---|---|---|
| `NU_Probabilidad_de_matricula__c` | `SF_PROB_FIELD` | PROBABILIDAD (0-100, entero) |
| `ProbabilityConfidence__c` | `SF_CONF_FIELD` | CONFIANZA (0-100, entero) |

La columna `CONFIANZA_ENV` se ha añadido a `PMAT_SF_SYNC_LOG` automáticamente
mediante `add_column_if_not_exists` en la primera ejecución tras la actualización.

---

## 4. Índice recomendado en PMAT_SF_SYNC_LOG

Para acelerar la consulta del último envío por oportunidad, crear índice:

```sql
CREATE INDEX IDX_SF_SYNC_OPP
ON PMATOWNER.PMAT_SF_SYNC_LOG (OPP_ID, FECHA_ENVIO DESC);
```

---

## 5. Índice recomendado en PMAT_PREDICTION

Para acelerar la vista `PMAT_PRED_ACTUAL` (que hace un `MAX(FECHA_INICIO_ETAPA)`
por `OPP_ID`):

```sql
CREATE INDEX IDX_PRED_OPP_FECHA
ON PMATOWNER.PMAT_PREDICTION (OPP_ID, FECHA_INICIO_ETAPA DESC);
```

---

## 6. Verificación operacional diaria (checklist)

```bash
# 1. Ver último log del cron
tail -50 logs/cron.log

# 2. Verificar que PMAT_PREDICTION se actualizó
python -c "
import sys; sys.path.insert(0, 'src')
from dotenv import load_dotenv; load_dotenv()
from oracle_connector import OracleConnector
import pandas as pd
conn = OracleConnector()
df = pd.DataFrame(conn.read_table('PMAT_PREDICTION'))
print('Total registros:', len(df))
print('Última actualización:', df['FECHA_ACTUALIZACION'].max())
"

# 3. Verificar write-back SF (últimas 100 filas del log)
python -c "
import sys; sys.path.insert(0, 'src')
from dotenv import load_dotenv; load_dotenv()
from oracle_connector import OracleConnector
import pandas as pd
conn = OracleConnector()
df = pd.DataFrame(conn.read_table('PMAT_SF_SYNC_LOG'))
df = df.sort_values('FECHA_ENVIO', ascending=False).head(100)
ok  = (df['STATUS'] == 'OK').sum()
err = (df['STATUS'] == 'ERROR').sum()
print(f'Últimos 100 envíos: {ok} OK, {err} ERROR')
if err:
    print(df[df['STATUS']=='ERROR'][['OPP_ID','DETALLE']].head(10).to_string())
"
```

---

## 7. Pendientes con el cliente

| # | Acción | Responsable | Estado |
|---|---|---|---|
| 1 | Confirmar nombre campo CONFIANZA en SF Opportunity | Cliente (Mario coordina) | ✅ Confirmado: `ProbabilityConfidence__c` |
| 2 | Confirmar que Connected App tiene permisos de escritura en Opportunity | Cliente | ✅ Verificado (write-back funcionando) |
| 3 | Solicitar a Oracle DBA crear job de purga mensual PMAT_SF_SYNC_LOG | Infraestructura | Pendiente |
| 4 | Crear índices IDX_SF_SYNC_OPP y IDX_PRED_OPP_FECHA | Infraestructura Oracle | Pendiente |

---

*Autor: Viewnext (Juan Velázquez y Mario Almendros)*
