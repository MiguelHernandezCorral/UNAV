# Guía de Despliegue a Producción

**Fecha de despliegue prevista:** 13 de abril de 2026
**Entorno destino:** Máquina virtual Linux de producción (nueva máquina)
**Responsable:** Juan Velázquez
**Credenciales Oracle pro:** proporcionadas por infraestructura
**Credenciales Salesforce pro:** proporcionadas por Usoa

---

## Resumen del pipeline

El pipeline realiza 4 fases:

| Fase | Módulo | Descripción |
|---|---|---|
| fase1 | `sf_extractor.py` | Extrae oportunidades activas de Salesforce → Oracle (`DATASET_BRUTO`) |
| fase2 | `cleaner.py` / `preprocessor.py` | Limpia y transforma los datos → Oracle (`DATASET_LIMPIO`) |
| fase3 | `predictor.py` | Aplica el modelo ML → Oracle (`PMAT_PREDICTION`) |
| fase4 | `sf_writer.py` | Envía probabilidades y confianza de vuelta a Salesforce |

**Campos que se escriben en Salesforce:**
- `NU_Probabilidad_de_matricula__c` — probabilidad de matrícula (0–100, entero)
- `ProbabilityConfidence__c` — confianza del modelo (0–100, entero)

---

## Paso 1 — Solicitar credenciales de producción

Antes del despliegue necesitas:

### Oracle producción
Solicitar a infraestructura:
- `ORA_HOST` — host del RAC de producción
- `ORA_PORT` — normalmente 1521
- `ORA_SERVICE` — service name de producción
- `ORA_USER` — usuario batch de producción (similar a PMAT_BTCH en pre)
- `ORA_SCHEMA` — esquema propietario (similar a PMATOWNER en pre)
- `ORA_PASSWORD` — contraseña

### Salesforce producción
Solicitar a Usoa:
- `SF_URL` — URL del org de producción (sin "sandbox" ni "fulladm")
- `SF_CLIENT_ID` — Consumer Key de la Connected App de producción
- `SF_CLIENT_SECRET` — Consumer Secret de la Connected App de producción

Los siguientes campos **no cambian** entre pre y pro:
```
SF_API_VERSION=60.0
SF_PROB_FIELD=NU_Probabilidad_de_matricula__c
SF_CONF_FIELD=ProbabilityConfidence__c
```

---

## Paso 2 — Preparar los modelos entrenados

Los modelos `.pkl` **no están en el repositorio** por su tamaño. Se encuentran en:
```
models/
├── modelo_grado.pkl
├── modelo_master.pkl
└── (cualquier otro .pkl)
```

Copia esos ficheros junto con el código para subirlos a la nueva MV.

---

## Paso 3 — Subir los ficheros a la nueva MV

Usa **FileZilla** para subir los archivos:

1. Conecta a la nueva MV vía SFTP.
2. Sube la carpeta `produccion_deploy/` completa a `/home/infra/jvelazquezc/UNAV/`.
3. Sube la carpeta `models/` a `/home/infra/jvelazquezc/UNAV/models/`.
4. **No subas** el fichero `.env` por FileZilla — créalo directamente en la MV (ver Paso 4).

---

## Paso 4 — Crear el fichero .env en la nueva MV

```bash
ssh <usuario>@<nueva-mv-pro>
cd /home/infra/jvelazquezc/UNAV
nano .env
```

Contenido del `.env` (sustituir los valores `<...>` con los reales):

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

Guarda con `Ctrl+O`, `Enter`, `Ctrl+X`.

Protege el fichero:
```bash
chmod 600 .env
```

---

## Paso 5 — Crear el entorno virtual e instalar dependencias

```bash
ssh <usuario>@<nueva-mv-pro>
cd /home/infra/jvelazquezc/UNAV
source .venv/bin/activate

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
chmod +x run_pipeline.sh
```

---

## Paso 6 — Verificar la instalación

```bash
ssh <usuario>@<nueva-mv-pro>
cd /home/infra/jvelazquezc/UNAV
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

Debe imprimir `OK — 3.x.x ...` sin errores.

---

## Paso 7 — Verificar conectividad Oracle

```bash
ssh <usuario>@<nueva-mv-pro>
cd /home/infra/jvelazquezc/UNAV
source .venv/bin/activate

nc -zv <ora-host-pro> 1521
```

Debe responder `Connection to <host> 1521 port [tcp/*] succeeded!`

---

## Paso 8 — Dry-run (sin escritura real)

```bash
ssh <usuario>@<nueva-mv-pro>
cd /home/infra/jvelazquezc/UNAV
source .venv/bin/activate

python src/pipeline.py --dry-run
```

Resultado esperado:
```
OK: 4 | Errores: 0
```

Si hay errores de conexión Oracle o Salesforce, revisar el `.env` y la conectividad de red.

---

## Paso 9 — Crear las tablas Oracle en producción

Si es la primera ejecución en el entorno de producción, las tablas Oracle se crean automáticamente al ejecutar el pipeline. No obstante, verificar con el DBA que el usuario batch tiene permisos `CREATE TABLE`, `INSERT`, `UPDATE`, `SELECT` sobre el esquema correspondiente.

Tablas que crea/usa el pipeline:
- `DATASET_BRUTO` — datos en bruto de SF
- `DATASET_LIMPIO` — datos limpios y transformados
- `PMAT_PREDICTION` — predicciones por oportunidad+etapa
- `PMAT_SF_SYNC_LOG` — log de envíos a Salesforce

---

## Paso 10 — Primera ejecución real

```bash
ssh <usuario>@<nueva-mv-pro>
cd /home/infra/jvelazquezc/UNAV
source .venv/bin/activate

bash run_pipeline.sh
```

Duración esperada: ~15 minutos (depende del volumen de datos de producción).

Verificar resultados:
```bash
python check_pipeline.py
```

---

## Paso 11 — Configurar ejecución automática (cron)

```bash
ssh <usuario>@<nueva-mv-pro>
cd /home/infra/jvelazquezc/UNAV
source .venv/bin/activate

bash setup_cron.sh
```

Esto añade una entrada en crontab para ejecutar el pipeline cada día a las 06:00.

Verificar:
```bash
crontab -l
```

---

## Diferencias clave entre Pre y Producción

| Aspecto | Pre (actual) | Pro (nuevo) |
|---|---|---|
| Datos | Copia de enero 2026 (~51.000 oportunidades) | Datos reales en tiempo real |
| Volumen | Estable (no crece) | Crecerá diariamente |
| Oracle host | `racdb-pre.si.unav.es` | A confirmar por infraestructura |
| SF URL | `https://unav--fulladm.sandbox.my.salesforce.com` | URL org producción |
| Modelos ML | Los mismos `.pkl` (entrenados sobre histórico UNAV) | Los mismos `.pkl` |
| Escritura SF | Solo sandbox | Salesforce real |

---

## Resolución de problemas frecuentes

### Error de conexión Oracle
```
ORA-12541: TNS:no listener
```
→ Verificar `ORA_HOST`, `ORA_PORT`, `ORA_SERVICE` en `.env`.
→ Verificar que la MV tiene acceso de red al RAC (`nc -zv <host> 1521`).

### Error de autenticación Salesforce
```
INVALID_CLIENT_ID / INVALID_CLIENT_SECRET
```
→ Verificar que `SF_CLIENT_ID` y `SF_CLIENT_SECRET` son los de la Connected App de producción, no del sandbox.

### Error en fase4 — registros rechazados por Salesforce
```
FIELD_CUSTOM_VALIDATION_EXCEPTION
```
→ Hay reglas de validación en SF que bloquean la escritura. Revisar con Usoa qué campos tienen validaciones.

### Pipeline lento en primera ejecución
Normal: en producción habrá más oportunidades que en pre. El pipeline procesa por lotes de 100 registros en fase4, con pausa de 0.5s entre lotes para respetar los límites de API de SF.

---

## Contactos

| Rol | Nombre | Para qué |
|---|---|---|
| Responsable técnico | Juan Velázquez | Dudas sobre el pipeline, errores técnicos |
| Infraestructura | — | Credenciales Oracle, permisos MV, conectividad red |
| Salesforce admin | Usoa | Credenciales SF, permisos, validaciones, campos |
