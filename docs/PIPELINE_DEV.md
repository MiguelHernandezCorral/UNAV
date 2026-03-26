# Pipeline UNAV — Guía de Despliegue en Linux DEV

## Resumen del flujo

```
Salesforce → Oracle (fase1)
         → DATASET_LIMPIO (fase2)
         → PMAT_PREDICTION + SHAP (fase4)
```

Cada fase escribe sus resultados en Oracle y genera logs en `logs/`.

---

## Estructura de despliegue

La carpeta `linux_deploy/` en la raíz del proyecto contiene todo lo necesario para la MV:

```
linux_deploy/
├── src/              ← código fuente (todos los .py)
├── models/           ← modelos .pkl (subir por FileZilla aparte)
├── docs/             ← documentación
├── logs/             ← directorio de logs (se crea al ejecutar)
├── requirements.txt
├── run_pipeline.sh
├── .env              ← credenciales VM (PMAT_BTCH, NO en git)
└── README.md         ← instrucciones de despliegue
```

### Sincronización Windows → linux_deploy/

Cuando se haga un cambio en el código, ejecutar desde la raíz del proyecto:
```batch
sync_deploy.bat
```
Este script copia `src/*.py`, `docs/*.md`, `requirements.txt` y `run_pipeline.sh` a `linux_deploy/`.
El fichero `linux_deploy/.env` **nunca se sobreescribe** (contiene credenciales de la MV).

### Credenciales por entorno

| Variable | Local (`.env`) | MV Linux (`linux_deploy/.env`) |
|---|---|---|
| ORA_USER | PMAT_USR | PMAT_BTCH |
| ORA_SCHEMA | PMATOWNER | PMATOWNER |
| ORA_HOST | racdb-pre.si.unav.es | racdb-pre.si.unav.es |
| ORA_SERVICE | UNSIDPRE.UNAV | UNSIDPRE.UNAV |
| ORA_PASSWORD | (local) | (MV) |

El código construye automáticamente el usuario proxy Oracle: `PMAT_BTCH[PMATOWNER]`.

> ⚠️ La contraseña de la MV contiene caracteres especiales (`)`  `>` `&`). En el fichero `.env` de la MV debe ir entre comillas simples:
> ```
> ORA_PASSWORD='w6IT%_)M>&'
> ```

---

## 1. Configuración del entorno Linux

### Entorno de DEV actual

| Parámetro | Valor |
|---|---|
| Servidor | `hydra4-pre.unav.es` |
| Usuario | `jvelazquezc` |
| Ruta proyecto | `/home/infra/jvelazquezc/UNAV` |
| Python | 3.9.25 (GCC 11.5.0, Red Hat) |
| Acceso | SSH (`ssh jvelazquezc@hydra4-pre.unav.es`) |
| Transferencia ficheros | FileZilla SFTP (puerto 22) |

### Crear el entorno virtual (primera vez)
```bash
cd /home/infra/jvelazquezc/UNAV
python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
chmod +x run_pipeline.sh
```

> La MV tiene internet directo a PyPI. No se necesitan wheels offline.

### Activar el entorno en sesiones posteriores
```bash
cd /home/infra/jvelazquezc/UNAV
source .venv/bin/activate
```

---

## 2. Variables de entorno (.env)

### Local (`.env`)
```env
ORA_HOST=racdb-pre.si.unav.es
ORA_PORT=1521
ORA_SERVICE=UNSIDPRE.UNAV
ORA_USER=PMAT_USR
ORA_SCHEMA=PMATOWNER
ORA_PASSWORD=<password_local>

SF_URL=https://unav--fulladm.sandbox.my.salesforce.com
SF_CLIENT_ID=<consumer_key>
SF_CLIENT_SECRET=<consumer_secret>
SF_API_VERSION=60.0
```

### MV Linux (`linux_deploy/.env`)
```env
ORA_HOST=racdb-pre.si.unav.es
ORA_PORT=1521
ORA_SERVICE=UNSIDPRE.UNAV
ORA_USER=PMAT_BTCH
ORA_SCHEMA=PMATOWNER
ORA_PASSWORD='w6IT%_)M>&'

SF_URL=https://unav--fulladm.sandbox.my.salesforce.com
SF_CLIENT_ID=<consumer_key>
SF_CLIENT_SECRET=<consumer_secret>
SF_API_VERSION=60.0
```

---

## 3. Ejecución manual

```bash
# Pipeline completo (todas las fases)
bash run_pipeline.sh

# Solo predicciones (asume que DATASET_LIMPIO ya existe)
bash run_pipeline.sh --phases fase4

# Ingesta + limpieza (sin predicciones)
bash run_pipeline.sh --phases fase1 fase2

# Limpieza + predicciones
bash run_pipeline.sh --phases fase2 fase4

# Dry-run: ejecuta sin escribir en Oracle ni Salesforce
bash run_pipeline.sh --dry-run

# Con historial completo de predicciones
bash run_pipeline.sh --phases fase4 --save-hist

# Continúa aunque falle una fase
bash run_pipeline.sh --no-stop-on-error

# Recrear DATASET_LIMPIO desde cero (cuando cambie el esquema)
source .venv/bin/activate
python src/cleaner.py --recreate
```

### Ejecución directa de Python (alternativa)
```bash
source .venv/bin/activate
python src/pipeline.py --phases fase4 --dry-run
```

---

## 4. Tabla de predicciones: PMAT_PREDICTION

La fase4 escribe en la tabla `PMATOWNER.PMAT_PREDICTION` con UPSERT inteligente:
solo actualiza registros cuando cambia la probabilidad.

| Columna | Tipo | Descripción |
|---|---|---|
| `OPP_ID_ETAPA_COMP` | PK | `OPP_ID__ETAPA__SUBETAPA` |
| `OPP_ID` | NVARCHAR2 | ID oportunidad Salesforce |
| `ETAPA` / `SUBETAPA` | NVARCHAR2 | Etapa del proceso |
| `TARGET_PRED` | NUMBER(1) | Predicción: 1=matrícula, 0=no |
| `TARGET_REAL` | NUMBER(1) | Resultado real (se rellena al cierre) |
| `PROBABILIDAD` | FLOAT | Probabilidad de matrícula [0–1] |
| `CONFIANZA` | FLOAT | Seguridad del modelo [0–1] |
| `MODELO` | NVARCHAR2 | Versión del modelo (`grado_v1`, etc.) |
| `EXPLICACION` | CLOB | JSON SHAP top-3 variables |
| `FECHA_PRED` | TIMESTAMP | Momento de la predicción |
| `FECHA_ACTUALIZACION` | TIMESTAMP | Última actualización del registro |

---

## 5. Variables del modelo y tratamiento de nulos

### Columnas protegidas del filtro >90% NA

Las siguientes columnas son features del modelo y **nunca se eliminan** aunque tengan
muchos nulos (es esperable al principio del proceso de admisión):

| Columna | Fuente SF | Motivo |
|---|---|---|
| `CU_IMPORTE_TOTAL` | `Opportunity.CU_Importe_total__c` | Importe total oportunidad |
| `NU_MEDIA_EXPEDIENTE_UNIVERSITA` | `Account.NU_Media_Expediente_Universitario__c` | Nota expediente universitario (Máster) |

Definidas en `src/cleaner.py::COLS_NUNCA_ELIMINAR`.

### Estrategia de imputación de nulos

| Variable | Estrategia | Motivo |
|---|---|---|
| Notas, importes, `PAID_PERCENT` | `fillna(0)` | Ausencia = valor cero es correcto |
| `FO_rentaFam_ges__c` | `fillna(media del segmento)` | Imputa por media para no sesgar |
| `CH_MATRICULA_SUJETA_BECA` | `fillna(0)` | Sin beca = 0 |
| Cualquier feature faltante del modelo | `fillna(0)` | Fallback de seguridad |

---

## 6. Programación con cron

```bash
crontab -e
```

Añadir línea (ejecuta el pipeline completo a las 06:00 cada día):
```cron
0 6 * * * /home/infra/jvelazquezc/UNAV/run_pipeline.sh >> /home/infra/jvelazquezc/UNAV/logs/cron.log 2>&1
```

Para solo predicciones (por ejemplo, diario a las 07:00):
```cron
0 7 * * * /home/infra/jvelazquezc/UNAV/run_pipeline.sh --phases fase4 >> /home/infra/jvelazquezc/UNAV/logs/cron_fase4.log 2>&1
```

---

## 7. Programación con systemd (recomendado para producción)

### Fichero de servicio: `/etc/systemd/system/unav-pipeline.service`
```ini
[Unit]
Description=Pipeline UNAV — Predicciones de matrícula
After=network.target

[Service]
Type=oneshot
User=jvelazquezc
WorkingDirectory=/home/infra/jvelazquezc/UNAV
ExecStart=/home/infra/jvelazquezc/UNAV/run_pipeline.sh
StandardOutput=append:/home/infra/jvelazquezc/UNAV/logs/systemd.log
StandardError=append:/home/infra/jvelazquezc/UNAV/logs/systemd.log
```

### Fichero de timer: `/etc/systemd/system/unav-pipeline.timer`
```ini
[Unit]
Description=Pipeline UNAV — Ejecución diaria

[Timer]
OnCalendar=*-*-* 06:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

### Activar el timer
```bash
sudo systemctl daemon-reload
sudo systemctl enable unav-pipeline.timer
sudo systemctl start unav-pipeline.timer

# Verificar estado
sudo systemctl status unav-pipeline.timer
sudo systemctl list-timers --all | grep unav
```

---

## 8. Logs y monitoreo

Los logs se guardan en `logs/`:

| Fichero | Contenido |
|---------|-----------|
| `logs/pipeline.log` | Log principal del orquestador (rotación diaria, 30 días) |
| `logs/pipeline_YYYYMMDD_HHMMSS.log` | Log de cada ejecución via `run_pipeline.sh` |
| `logs/sf_extract_all.log` | Log detallado de la ingesta Salesforce |
| `logs/cron.log` | Salida del cron (si se usa) |

### Ver últimas ejecuciones
```bash
tail -100 logs/pipeline.log
tail -f logs/pipeline.log   # en tiempo real
ls -lt logs/pipeline_*.log | head -5
```

### Verificar que PMAT_PREDICTION se actualizó
```bash
python -c "
import sys; sys.path.insert(0, 'src')
from dotenv import load_dotenv; load_dotenv()
from oracle_connector import OracleConnector
import pandas as pd
conn = OracleConnector()
df = pd.DataFrame(conn.read_table('PMAT_PREDICTION'))
print('Total registros:', len(df))
print('Última actualización:', df['FECHA_ACTUALIZACION'].max())
print('Predicciones matrícula:', df['TARGET_PRED'].sum())
"
```

---

## 9. Fases del pipeline

| Fase | Módulo | Descripción |
|------|--------|-------------|
| `fase1` | `sf_extract_all.py` | Ingesta de 10 entidades Salesforce → Oracle |
| `fase2` | `cleaner.py` | Limpieza de datos → tabla `DATASET_LIMPIO` (truncate + insert) |
| `fase4` | `predictor.py` | Predicciones con PyCaret → `PMAT_PREDICTION` (UPSERT por PK) |

---

## 10. requirements.txt (Python 3.9)

```
# Core ML (versiones exactas — no cambiar)
numpy==1.26.4
pandas==2.1.4
scipy==1.11.4
scikit-learn==1.4.2
pycaret==3.3.2
lightgbm==4.6.0
shap==0.49.1

# Oracle, HTTP, entorno (última versión compatible con Python 3.9)
oracledb
requests
python-dotenv
```

> `shap==0.49.1` es la última versión compatible con Python 3.9 (0.50.0+ requiere Python 3.11).

---

## 11. Troubleshooting

### Error: "No module named pandas" al ejecutar pipeline
```bash
# El venv no está activado
source /home/infra/jvelazquezc/UNAV/.venv/bin/activate
```

### Error: "invalid option name: set: pipefail" en run_pipeline.sh
```bash
# Saltos de línea Windows (CRLF) en el script
sed -i 's/\r//' run_pipeline.sh
```

### Error: "syntax error near unexpected token ')'" en .env
```bash
# Contraseña con caracteres especiales — ponerla entre comillas simples
nano .env
# ORA_PASSWORD='w6IT%_)M>&'
```

### Error: ORA-00904 "columna": invalid identifier al recrear tabla
```bash
# El esquema de DATASET_LIMPIO cambió — recrear la tabla
python src/cleaner.py --recreate
```

### Error de conexión Oracle "Connection refused"
- Verificar que el puerto 1521 está abierto desde la MV:
```bash
python -c "import socket; s=socket.socket(); s.settimeout(5); s.connect(('racdb-pre.si.unav.es',1521)); print('OK')"
```
- Si falla: solicitar a infraestructura abrir puerto TCP 1521 desde `hydra4-pre.unav.es` hacia `racdb-pre.si.unav.es`

### Error de autenticación Salesforce
- Verificar que el Connected App tiene `client_credentials` habilitado
- Comprobar que `SF_CLIENT_ID` y `SF_CLIENT_SECRET` son correctos

---

*Autor: Viewnext (Juan Velázquez y Mario Almendros)*
