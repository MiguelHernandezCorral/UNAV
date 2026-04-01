# Verificación de producción — Guía para el cliente

**Objetivo:** Comprobar que el pipeline funciona correctamente en el entorno de producción **sin modificar ni ejecutar nada que escriba datos reales**.

**Responsable técnico de dudas:** Juan Velázquez

---

## Requisitos previos (antes de empezar)

Necesitas tener listos:

- Acceso SSH a la MV de producción
- Acceso SFTP (FileZilla u similar) a la MV de producción
- Credenciales Oracle de producción: host, puerto, service name, usuario, contraseña
- Credenciales Salesforce de producción: URL org, Client ID, Client Secret
- Los ficheros del pipeline (carpeta `produccion_deploy/`) — los proporciona Juan
- Los modelos ML (carpeta `models/`) — los proporciona Juan

---

## BLOQUE 1 — Subir los ficheros

**Usando FileZilla (o similar SFTP):**

1. Conecta a la MV por SFTP
2. Sube la carpeta `produccion_deploy/` a `~/UNAV/`
3. Sube la carpeta `models/` a `~/UNAV/models/`

Estructura resultante en la MV:
```
~/UNAV/
├── src/
├── models/
│   ├── modelo_grado.pkl
│   └── modelo_master.pkl
├── requirements.txt
├── run_pipeline.sh
└── setup_cron.sh
```

**No subas el fichero `.env`** — se crea directamente en la MV en el siguiente paso.

---

## BLOQUE 2 — Crear el fichero de credenciales

```bash
ssh <usuario>@<mv-pro>
cd ~/UNAV
nano .env
```

Pega este contenido y rellena los valores `<...>` con las credenciales reales:

```dotenv
# Oracle — Producción
ORA_HOST=<host-oracle-pro>
ORA_PORT=1521
ORA_SERVICE=<service-pro>
ORA_USER=<usuario-batch-pro>
ORA_SCHEMA=<esquema-pro>
ORA_PASSWORD=<password-pro>

# Salesforce — Producción
SF_URL=<url-org-pro>
SF_SITE=
SF_CLIENT_ID=<client-id-pro>
SF_CLIENT_SECRET=<client-secret-pro>
SF_API_VERSION=60.0
SF_PROB_FIELD=NU_Probabilidad_de_matricula__c
SF_CONF_FIELD=ProbabilityConfidence__c
```

> **Importante:** No pongas comentarios al final de ninguna línea de valor. Solo en líneas propias que empiecen por `#`.

Guarda: `Ctrl+O` → `Enter` → `Ctrl+X`

Protege el fichero:
```bash
chmod 600 .env
```

---

## BLOQUE 3 — Preparar el entorno Python

```bash
cd ~/UNAV
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
chmod +x run_pipeline.sh
```

Resultado esperado: instalación sin errores. Puede tardar unos minutos.

---

## BLOQUE 4 — Verificaciones (solo lectura, nada se modifica)

Ejecuta cada comprobación en orden. **Ninguna escribe datos.**

### 4.1 — Verificar que las librerías están instaladas

```bash
source .venv/bin/activate
python -c "
import sys; sys.path.insert(0, 'src')
from pipeline import run_pipeline, PHASE_REGISTRY
from oracle_connector import OracleConnector
from predictor import run_predictions_v2
from sf_writer import SFWriter
print('OK —', sys.version)
"
```

**Resultado esperado:** `OK — 3.x.x ...`
Si hay `ModuleNotFoundError`, avisar a Juan.

---

### 4.2 — Verificar conectividad de red con Oracle

```bash
nc -zv <ora-host-pro> 1521
```

**Resultado esperado:** `Connection to <host> 1521 port [tcp/*] succeeded!`
Si falla, la MV no tiene acceso de red al RAC — contactar con infraestructura.

---

### 4.3 — Verificar autenticación Oracle

```bash
python -c "
import sys, os; sys.path.insert(0, 'src')
from dotenv import load_dotenv; load_dotenv()
from oracle_connector import OracleConnector
conn = OracleConnector()
print('Conexión Oracle OK')
conn.close()
"
```

**Resultado esperado:** `Conexión Oracle OK`

Errores frecuentes:
- `ORA-01017` → credenciales incorrectas en `.env`
- `ORA-12541` → host/puerto/service name incorrectos en `.env`
- `DPY-4018` → hay un comentario inline en la línea `ORA_*` del `.env` — eliminarlo

---

### 4.4 — Dry-run del pipeline completo

Este paso **simula** la ejecución completa sin escribir nada en Oracle ni en Salesforce:

```bash
python src/pipeline.py --dry-run
```

**Resultado esperado:**
```
OK: 4 | Errores: 0
```

Si alguna fase falla, el dry-run lo indica con el nombre de la fase y el error. Enviarlo a Juan.

---

### 4.5 — Verificar estado general

```bash
python check_pipeline.py
```

Muestra el estado de las tablas Oracle y la última ejecución registrada.

---

## BLOQUE 5 — Checklist de validación

Marca cada punto antes de dar el visto bueno:

- [ ] Ficheros subidos correctamente a `~/UNAV/`
- [ ] Modelos `.pkl` en `~/UNAV/models/`
- [ ] `.env` creado con credenciales de producción, sin comentarios inline
- [ ] `chmod 600 .env` aplicado
- [ ] Entorno virtual creado y dependencias instaladas sin errores
- [ ] **4.1** — Librerías: `OK`
- [ ] **4.2** — Conectividad red Oracle: `succeeded`
- [ ] **4.3** — Autenticación Oracle: `Conexión Oracle OK`
- [ ] **4.4** — Dry-run: `OK: 4 | Errores: 0`
- [ ] **4.5** — `check_pipeline.py` sin errores

---

## Si todo está OK — siguiente paso

Con el checklist completo, comunicar a Juan que el entorno está validado. Él decidirá cuándo lanzar la primera ejecución real (`bash run_pipeline.sh`) y activar el cron.

**No ejecutar `run_pipeline.sh` ni `setup_cron.sh` sin confirmación de Juan.**

---

## Contacto para incidencias

| Problema | A quién avisar |
|---|---|
| Error en librerías / dry-run | Juan Velázquez |
| Error de red / acceso MV | Infraestructura |
| Error credenciales Salesforce | Usoa |
| Error credenciales Oracle | Infraestructura |
