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
2. Sube la carpeta `produccion_deploy/` completa a `~/UNAV/`.
3. Sube la carpeta `models/` a `~/UNAV/models/`.
4. **No subas** el fichero `.env` por FileZilla — créalo directamente en la MV (ver Paso 4).

---

## Paso 4 — Crear el fichero .env en la nueva MV

```bash
ssh <usuario>@<nueva-mv-pro>
cd ~/UNAV
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
cd ~/UNAV
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
cd ~/UNAV
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
cd ~/UNAV
source .venv/bin/activate

nc -zv <ora-host-pro> 1521
```

Debe responder `Connection to <host> 1521 port [tcp/*] succeeded!`

---

## Paso 8 — Dry-run (sin escritura real)

```bash
ssh <usuario>@<nueva-mv-pro>
cd ~/UNAV
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
cd ~/UNAV
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
cd ~/UNAV
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

---

## Parches aplicados — historial para PRO

Esta sección recoge los cambios técnicos que hay que aplicar en PRO a medida que se validan en PRE. Cada parche indica qué archivos tocar y si hace falta recrear tablas Oracle.

---

### Parche 1 — Fix clasificación Grado/Máster por RECORDTYPEID
**Fecha validación PRE:** 07/05/2026
**Archivos modificados:**
- `src/preprocessor.py`
- `src/cleaner.py`

**Problema:** La clasificación de oportunidades entre Grado y Máster usaba el campo `TITULACION`, que contiene "Máster" con tilde. El filtro `str.contains("MASTER")` no la reconocía y todas las oportunidades de Máster se procesaban como Grado, generando predicciones incorrectas.

**Fix aplicado:** Se usa ahora `RECORDTYPEID` (IDs exactos de RecordType de Salesforce) como clasificador principal:
- Grado: `012w0000000K4QPAA0`, `012w0000000K4QTAA0`
- Máster: `012w0000000K4QUAA0`, `012w0000000K4QQAA0`

`RECORDTYPEID` y `RECORDTYPENAME` se añadieron a `COLUMNAS_FINALES` en `cleaner.py` para que lleguen a `DATASET_LIMPIO`, y a `VARS_EXCLUIR` en `preprocessor.py` para que no entren como features del modelo.

**Pasos para aplicar en PRO:**

```bash
ssh <usuario>@<mv-pro>
cd ~/UNAV
source .venv/bin/activate

# 1. Subir los archivos actualizados por FileZilla:
#    produccion_deploy/src/cleaner.py   → ~/UNAV/src/cleaner.py
#    produccion_deploy/src/preprocessor.py → ~/UNAV/src/preprocessor.py

# 2. Recrear DATASET_LIMPIO (schema cambia: añade columnas RECORDTYPEID y RECORDTYPENAME)
python src/cleaner.py --recreate

# 3. Relanzar fases 3 y 4
python src/predictor.py

# 4. Verificar conteos grado/máster
python -c "
import sys; sys.path.insert(0, 'src')
from oracle_connector import OracleConnector
conn = OracleConnector()
cur = conn.conn.cursor()
cur.execute('SELECT MODELO, COUNT(DISTINCT OPP_ID) N FROM PMAT_PREDICTION GROUP BY MODELO ORDER BY MODELO')
for r in cur.fetchall(): print(r)
cur.close()
"
# Resultado esperado (proporcional a volumen PRO):
#   ('grado_v1', XXXX)
#   ('master_v1', XXXX)   ← debe tener registros, no estar vacío
```

---

### Parche 2 — Fix variable PAID_PERCENT (columna duplicada)
**Fecha validación PRE:** 07/05/2026
**Archivos modificados:**
- `src/cleaner.py`

**Problema:** `DATASET_LIMPIO` guardaba dos columnas equivalentes al porcentaje pagado:
- `PAID_PERCENT` — campo directo de Salesforce (`PAID_PERCENT__C`), casi siempre vacío para prospectos nuevos.
- `PORCENTAJE_PAGADO_FINAL` — valor calculado desde ECBS (precio aplicado / precio ordinario × 100), con datos reales.

El preprocessor renombra `PORCENTAJE_PAGADO_FINAL` → `PAID_PERCENT` al leer Oracle. Esto creaba una columna duplicada y el pipeline se quedaba con la versión SF (vacía), que tras imputación quedaba a 0 para todos → se filtraba como constante y el modelo perdía la variable.

**Fix aplicado:** Se eliminó `PAID_PERCENT` de `COLUMNAS_FINALES` en `cleaner.py`. Solo se guarda `PORCENTAJE_PAGADO_FINAL` (el calculado), que el preprocessor ya renombra correctamente.

**Pasos para aplicar en PRO:**

```bash
ssh <usuario>@<mv-pro>
cd ~/UNAV
source .venv/bin/activate

# 1. Subir el archivo actualizado por FileZilla:
#    produccion_deploy/src/cleaner.py   → ~/UNAV/src/cleaner.py

# 2. Recrear DATASET_LIMPIO (schema cambia: elimina columna PAID_PERCENT del SF)
python src/cleaner.py --recreate

# 3. Verificar que PORCENTAJE_PAGADO_FINAL tiene valores no constantes
python -c "
import sys; sys.path.insert(0, 'src')
from oracle_connector import OracleConnector
conn = OracleConnector()
cur = conn.conn.cursor()
cur.execute('SELECT COUNT(*) TOTAL, COUNT(PORCENTAJE_PAGADO_FINAL) CON_VALOR, ROUND(AVG(PORCENTAJE_PAGADO_FINAL),2) MEDIA FROM DATASET_LIMPIO')
for r in cur.fetchall(): print(r)
cur.close()
"
# CON_VALOR debe ser > 0 y MEDIA no debe ser exactamente 0.0

# 4. Relanzar fases 3 y 4
python src/predictor.py
```

> **Nota:** Los parches 1 y 2 se pueden aplicar juntos en un solo `--recreate` si se suben los dos archivos antes de ejecutarlo.

---

### Parche 3 — Extracción y predicciones para ambos cursos académicos
**Fecha validación PRE:** 07/05/2026
**Archivos modificados:**
- `src/sf_extract_all.py`
- `src/pipeline.py`
- `src/predictor.py`
- `src/cleaner.py`

**Problema:** La pipeline solo extraía y procesaba el curso 2026/2027. El curso 2025/2026 no se incluía, por lo que sus oportunidades no recibían predicciones actualizadas.

**Fix aplicado:**
- La extracción SF ahora obtiene ambos cursos (2025/2026 y 2026/2027) por defecto en cada ejecución.
- Se añadió soporte para lanzar un curso concreto con el flag `--curso` (útil para backfill o depuración).
- `FUENTE_DATOS` en `DATASET_LIMPIO` se asigna dinámicamente: `SF_2026_27` o `SF_2025_26` según el curso de cada oportunidad (antes era siempre `SF_2026_27`).
- La lógica de separación grado/máster en el predictor se actualizó para incluir ambos cursos correctamente.

**Pasos para aplicar en PRO:**

```bash
ssh <usuario>@<mv-pro>
cd ~/UNAV
source .venv/bin/activate

# 1. Subir los archivos actualizados por FileZilla:
#    produccion_deploy/src/sf_extract_all.py → ~/UNAV/src/sf_extract_all.py
#    produccion_deploy/src/pipeline.py       → ~/UNAV/src/pipeline.py
#    produccion_deploy/src/predictor.py      → ~/UNAV/src/predictor.py
#    produccion_deploy/src/cleaner.py        → ~/UNAV/src/cleaner.py

# 2. Relanzar la pipeline completa (extrae ambos cursos desde SF)
python src/pipeline.py --fase1 --fase2 --fase3 --fase4

# 3. Verificar predicciones por curso
python -c "
import sys; sys.path.insert(0, 'src')
from oracle_connector import OracleConnector
conn = OracleConnector()
cur = conn.conn.cursor()
cur.execute('''
    SELECT FUENTE_DATOS, MODELO, COUNT(DISTINCT OPP_ID) N
    FROM PMAT_PREDICTION
    GROUP BY FUENTE_DATOS, MODELO
    ORDER BY FUENTE_DATOS, MODELO
''')
for r in cur.fetchall(): print(r)
cur.close()
"
# Resultado esperado: filas para SF_2025_26 y SF_2026_27, ambos con grado_v1 y master_v1
```

---

### Parche 4 — Robustez en extracción SF y carga Oracle
**Fecha validación PRE:** 07/05/2026
**Archivos modificados:**
- `src/sf_extractor.py`
- `src/oracle_connector.py`

**Problemas resueltos:**

| Error | Causa | Fix |
|---|---|---|
| Timeout en `activity_history` | `timeout=60` insuficiente para ~35.000 registros paginados | Timeout ampliado a 300 s con 3 reintentos automáticos (backoff 30/60/90 s) |
| `ORA-22835` en ACCOUNT | Inferidor de tipos saltaba de NVARCHAR2(2000) a CLOB para strings de 2001–4000 chars, causando conflicto de tipo al hacer bind | Añadido nivel NVARCHAR2(4000) antes de CLOB |
| `ORA-00904` en CASES / ACTIVITY_HISTORY | Columnas del curso 2025/2026 no existían en Oracle (tablas creadas solo con datos 2026/2027) | El upsert detecta y añade automáticamente columnas nuevas antes del MERGE |
| `ORA-12899` en ACCOUNT (y otras tablas) | Strings del curso 2025/2026 más largos que el límite de columnas creadas con 2026/2027 | Antes del MERGE se consulta `all_tab_columns` y se truncan silenciosamente los valores que superen el `char_length` real de cada columna |

**Pasos para aplicar en PRO:**

```bash
ssh <usuario>@<mv-pro>
cd ~/UNAV
source .venv/bin/activate

# 1. Subir los archivos actualizados por FileZilla:
#    produccion_deploy/src/sf_extractor.py      → ~/UNAV/src/sf_extractor.py
#    produccion_deploy/src/oracle_connector.py  → ~/UNAV/src/oracle_connector.py

# No requiere recrear tablas Oracle.
# El fix se aplica automáticamente en la siguiente ejecución del pipeline.
```

> **Acción DBA requerida antes de lanzar en PRO:** El tablespace que aloja las tablas del pipeline debe tener espacio suficiente para almacenar datos de dos cursos académicos. En PRE fue necesario ampliar `PMATDA00`. Confirmar con infraestructura que el tablespace de PRO tiene espacio disponible (o AUTOEXTEND activado) antes de la primera ejecución completa.

---

### Orden de aplicación recomendado

Si se van a aplicar todos los parches a la vez (despliegue inicial), el orden es:

1. Subir **todos** los archivos de `produccion_deploy/src/` de una vez.
2. Confirmar con el DBA que el tablespace tiene espacio suficiente.
3. Ejecutar:
```bash
python src/cleaner.py --recreate   # aplica parches 1, 2 y 3 en DATASET_LIMPIO
python src/pipeline.py --fase1 --fase2 --fase3 --fase4  # ejecución completa
```
