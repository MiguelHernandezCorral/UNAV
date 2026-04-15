# Cómo lanzar el pipeline — Pre y Producción

## Requisitos previos

- Acceso SSH a la MV correspondiente
- Acceso SFTP (FileZilla) para subir archivos
- Fichero `.env` con credenciales del entorno
- Modelos `.pkl` en `~/UNAV/models/`

---

## Entornos

| Entorno | MV | Usuario | Ruta |
|---|---|---|---|
| Preproducción | hydra4-pre.unav.es | jvelazquezc | ~/UNAV |
| Producción | (host pro) | probmatr | ~/UNAV |

---

## Conectar por SSH

### Preproducción
```bash
ssh jvelazquezc@hydra4-pre.unav.es
```

### Producción
```bash
ssh probmatr@<host-pro>
```

---

## Primera instalación en una MV nueva

### 1 — Subir ficheros (FileZilla)

Conectar por SFTP al host correspondiente y subir:
- Carpeta `produccion_deploy/` completa → `~/UNAV/`
- Carpeta `models/` → `~/UNAV/models/`

No subir el `.env` — crearlo directamente en la MV (ver paso 2).

### 2 — Crear el .env

```bash
cd ~/UNAV
nano .env
```

Contenido:
```dotenv
ORA_HOST=<host-oracle>
ORA_PORT=1521
ORA_SERVICE=<service>
ORA_USER=<usuario-batch>
ORA_SCHEMA=<esquema>
ORA_PASSWORD=<password>

SF_URL=https://<org>.my.salesforce.com
SF_SITE=
SF_CLIENT_ID=<client-id>
SF_CLIENT_SECRET=<client-secret>
SF_API_VERSION=60.0
SF_PROB_FIELD=NU_Probabilidad_de_matricula__c
SF_CONF_FIELD=ProbabilityConfidence__c
```

> **Importante:** El valor de `SF_URL` debe incluir `https://`. No añadir comentarios en la misma línea que un valor.

Guardar: `Ctrl+O` → `Enter` → `Ctrl+X`

```bash
chmod 600 .env
```

### 3 — Crear entorno virtual e instalar dependencias

```bash
cd ~/UNAV
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
chmod +x run_pipeline.sh
```

---

## Lanzar el pipeline

### Pipeline completo

```bash
cd ~/UNAV
source .venv/bin/activate
bash run_pipeline.sh
```

### Fases concretas

```bash
# Solo ingesta y limpieza (sin predicciones ni SF)
python src/pipeline.py --phases fase1 fase2

# Ingesta, limpieza y predicciones (sin write-back a SF)
python src/pipeline.py --phases fase1 fase2 fase3

# Solo write-back a Salesforce (si fases 1-3 ya están OK)
python src/pipeline.py --phases fase4

# Dry-run — simula sin escribir nada
python src/pipeline.py --dry-run
```

### Con historial de predicciones

```bash
python src/pipeline.py --phases fase3 --save-hist
```
Guarda también en `PMAT_PREDICTION_HIST` para auditoría.

---

## Verificar resultados

### Estado general

```bash
cd ~/UNAV && source .venv/bin/activate
python check_pipeline.py
```

### Verificar los dos cursos en DATASET_BRUTO

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from dotenv import load_dotenv; load_dotenv()
from oracle_connector import OracleConnector
import pandas as pd
conn = OracleConnector()
df = pd.DataFrame(conn.read_table('OPPORTUNITY'))
print(df['PL_CURSO_ACADEMICO__C'].value_counts())
"
```

### Verificar predicciones

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from dotenv import load_dotenv; load_dotenv()
from oracle_connector import OracleConnector
import pandas as pd
conn = OracleConnector()
df = pd.DataFrame(conn.read_table('PMAT_PREDICTION'))
print('Total predicciones:', len(df))
print('Última actualización:', df['FECHA_ACTUALIZACION'].max())
print(df['MODELO'].value_counts())
"
```

### Verificar write-back a Salesforce

```bash
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
"
```

---

## Cron diario (ejecución automática)

### Instalar

```bash
cd ~/UNAV && source .venv/bin/activate
bash setup_cron.sh
```

### Verificar

```bash
crontab -l
```

La entrada instalada ejecuta el pipeline cada día a las 03:00:
```
0 3 * * * ~/UNAV/run_pipeline.sh >> ~/UNAV/logs/cron.log 2>&1
```

### Seguir logs en tiempo real

```bash
tail -f ~/UNAV/logs/cron.log
tail -f ~/UNAV/logs/pipeline.log
```

---

## Errores frecuentes y soluciones

| Error | Causa | Solución |
|---|---|---|
| `DPY-4027: no configuration directory` | DSN sin `https://` o sin `//` | Añadir `https://` a `SF_URL` o `//` al DSN Oracle |
| `DPY-4018: cannot parse connect string` | Comentario inline en `.env` | Eliminar el comentario de la misma línea del valor |
| `ORA-00942: table or view does not exist` | `read_table` sin prefijo de esquema | Verificar versión de `oracle_connector.py` — debe usar `schema.table` |
| `ORA-01017: invalid username/password` | Credenciales incorrectas | Revisar `ORA_USER`, `ORA_SCHEMA`, `ORA_PASSWORD` en `.env` |
| `ORA-12541: TNS no listener` | Host/puerto/service incorrecto | Revisar `ORA_HOST`, `ORA_PORT`, `ORA_SERVICE` |
| `ORA-12899: value too large` | Campo DETALLE demasiado largo | Verificar versión de `sf_writer.py` — debe truncar a 100 chars |
| `MissingSchema: Invalid URL` | `SF_URL` sin `https://` | Añadir `https://` al valor de `SF_URL` en `.env` |
| `ModuleNotFoundError: oracle_connector` | `sys.path` antes del import | Añadir `sys.path.insert` antes de cualquier import local |
| `'DataFrame' has no attribute 'dtype'` | Columnas duplicadas en el DataFrame | Verificar versión de `preprocessor.py` — debe deduplicar columnas |
