# Pipeline UNAV — Guía de Despliegue en Linux DEV

## Resumen del flujo

```
Salesforce → Oracle (fase1)
         → DATASET_LIMPIO (fase2)
         → Validación (validate)
         → PREDICCIONES_V2 + SHAP (fase4)
```

Cada fase escribe sus resultados en Oracle y genera logs en `logs/`.

---

## 1. Configuración del entorno Linux

### Requisitos del sistema
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev \
                        libgomp1 build-essential git
```

> `libgomp1` es necesario para LightGBM (que usa OpenMP internamente).

### Python via pyenv (alternativa si el sistema tiene Python < 3.11)
```bash
curl https://pyenv.run | bash
# Añadir pyenv al PATH en ~/.bashrc:
#   export PYENV_ROOT="$HOME/.pyenv"
#   export PATH="$PYENV_ROOT/bin:$PATH"
#   eval "$(pyenv init -)"
source ~/.bashrc

pyenv install 3.11.9
cd /ruta/a/UNAV
pyenv local 3.11.9
```

### Crear el entorno virtual
```bash
cd /ruta/a/UNAV
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Variables de entorno (.env)

Crea el fichero `.env` en la raíz del proyecto (nunca se sube a git):

```env
# Oracle
ORA_HOST=<hostname_oracle>
ORA_PORT=1521
ORA_SERVICE=<service_name>
ORA_USER=PMAT_USR
ORA_SCHEMA=PMATOWNER
ORA_PASSWORD=<password>

# Salesforce
SF_URL=https://<instancia>.salesforce.com
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

# Dry-run: ejecuta sin escribir en Oracle ni Salesforce
bash run_pipeline.sh --dry-run

# Con historial completo de predicciones
bash run_pipeline.sh --phases fase4 --save-hist

# Continúa aunque falle una fase
bash run_pipeline.sh --no-stop-on-error
```

### Ejecución directa de Python (alternativa)
```bash
source .venv/bin/activate
python src/pipeline.py --phases fase4 --dry-run
```

---

## 4. Programación con cron

```bash
crontab -e
```

Añadir línea (ejecuta el pipeline completo a las 06:00 cada día):
```cron
0 6 * * * /ruta/a/UNAV/run_pipeline.sh >> /ruta/a/UNAV/logs/cron.log 2>&1
```

Para solo predicciones (por ejemplo, diario a las 07:00):
```cron
0 7 * * * /ruta/a/UNAV/run_pipeline.sh --phases fase4 >> /ruta/a/UNAV/logs/cron_fase4.log 2>&1
```

---

## 5. Programación con systemd (recomendado para producción)

### Fichero de servicio: `/etc/systemd/system/unav-pipeline.service`
```ini
[Unit]
Description=Pipeline UNAV — Predicciones de matrícula
After=network.target

[Service]
Type=oneshot
User=<usuario_linux>
WorkingDirectory=/ruta/a/UNAV
ExecStart=/ruta/a/UNAV/run_pipeline.sh
StandardOutput=append:/ruta/a/UNAV/logs/systemd.log
StandardError=append:/ruta/a/UNAV/logs/systemd.log
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

## 6. Logs y monitoreo

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
ls -lt logs/pipeline_*.log | head -5
```

### Verificar que PREDICCIONES_V2 se actualizó
```sql
SELECT COUNT(*), MAX(FECHA_ACTUALIZACION)
FROM PREDICCIONES_V2;
```

---

## 7. Fases del pipeline

| Fase | Módulo | Descripción |
|------|--------|-------------|
| `fase1` | `sf_extract_all.py` | Ingesta de 10 entidades Salesforce → Oracle |
| `fase2` | `cleaner.py` | Limpieza de datos → tabla `DATASET_LIMPIO` |
| `validate` | `validator.py` | Validación de calidad y discrepancias |
| `fase4` | `predictor.py` | Predicciones con PyCaret → `PREDICCIONES_V2` |

El módulo `validator` bloquea la ejecución si detecta errores P1 (críticos).
Ver `docs/VALIDADOR.md` para el detalle de checks y prioridades.

---

## 8. Troubleshooting

### Error: "No module named pycaret"
```bash
source .venv/bin/activate
pip install pycaret[full]
```

### Error: "libgomp.so not found" (LightGBM)
```bash
sudo apt-get install libgomp1
```

### Error de conexión Oracle
- Verificar que `.env` contiene las credenciales correctas
- Verificar conectividad: `nc -zv ORA_HOST 1521`
- El cliente Oracle Thin (oracledb) no requiere Oracle Client instalado

### Error de autenticación Salesforce
- Verificar que el Connected App tiene `client_credentials` habilitado
- Comprobar que `SF_CLIENT_ID` y `SF_CLIENT_SECRET` son correctos
