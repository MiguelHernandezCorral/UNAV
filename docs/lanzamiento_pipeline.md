# Guía de Lanzamiento de la Pipeline

**Responsable:** Juan Velázquez / Mario Almendros  
**Proyecto:** Predicción de Matrícula UNAV  
**Acceso externo SSH:** `external.unav.es`

Esta guía cubre cómo lanzar la pipeline en **local (Windows)**, en **preproducción (MV Linux)** y en **producción**, incluyendo verificación de resultados y resolución de los errores más frecuentes.

---

## Índice

1. [Requisitos previos](#1-requisitos-previos)
2. [Lanzar en local (Windows)](#2-lanzar-en-local-windows)
3. [Subir archivos a la MV con FileZilla](#3-subir-archivos-a-la-mv-con-filezilla)
4. [Conectarse a la MV por SSH desde PowerShell](#4-conectarse-a-la-mv-por-ssh-desde-powershell)
5. [Lanzar en preproducción](#5-lanzar-en-preproducción)
6. [Lanzar en producción](#6-lanzar-en-producción)
7. [Verificar que la pipeline ha funcionado](#7-verificar-que-la-pipeline-ha-funcionado)
8. [Guía de errores frecuentes](#8-guía-de-errores-frecuentes)

---

## 1. Requisitos previos

| Requisito | Local | Pre | Pro |
|---|---|---|---|
| Python 3.10+ | ✅ | ✅ (ya instalado en MV) | ✅ |
| Fichero `.env` con credenciales | ✅ | ✅ | ✅ |
| Modelos `.pkl` en `models/` | ✅ | ✅ | ✅ |
| Entorno virtual `.venv` activado | ✅ | ✅ | ✅ |
| Acceso de red a Oracle | ✅ (VPN si es desde fuera) | ✅ | ✅ |
| Acceso a Salesforce | ✅ | ✅ (sandbox) | ✅ (pro) |

### Credenciales necesarias en el `.env`

```dotenv
SF_URL=https://unav--fulladm.sandbox.my.salesforce.com   # pre; en pro usar URL org producción
SF_SITE=
SF_CLIENT_ID=<client-id>
SF_CLIENT_SECRET=<client-secret>
SF_API_VERSION=60.0
SF_PROB_FIELD=NU_Probabilidad_de_matricula__c
SF_CONF_FIELD=ProbabilityConfidence__c

ORA_HOST=racdb-pre.si.unav.es    # pre; en pro el host del RAC de producción
ORA_PORT=1521
ORA_SERVICE=UNSIDPRE.UNAV        # pre; en pro el service name de producción
ORA_USER=PMAT_USR
ORA_SCHEMA=PMATOWNER
ORA_PASSWORD=<password>
```

> **Importante:** El valor de `SF_URL` debe incluir `https://`. No poner comentarios en la misma línea que un valor — rompe la lectura del `.env`.

---

## 2. Lanzar en local (Windows)

### Paso 1 — Activar el entorno virtual

Abrir PowerShell o CMD en la carpeta raíz del proyecto:

```powershell
cd C:\Users\jvelazquezc\Desktop\UNAV
.venv\Scripts\activate
```

El prompt cambia a `(.venv)` cuando está activo.

### Paso 2 — Verificar que el `.env` está presente

```powershell
dir .env
```

Si no existe, copiarlo desde `.env.example` y rellenar los valores.

### Paso 3 — Verificar que los modelos están presentes

```powershell
dir models\
```

Deben aparecer `modelo_final_grado.pkl` y `modelo_final_master.pkl`.

### Paso 4 — Ejecutar la pipeline

```powershell
# Pipeline completo (todas las fases)
python src/pipeline.py

# Dry-run — simula todo sin escribir nada en BBDD ni en SF
python src/pipeline.py --dry-run

# Solo una fase concreta
python src/pipeline.py --phases fase1
python src/pipeline.py --phases fase1 fase2
python src/pipeline.py --phases fase3
python src/pipeline.py --phases fase4

# Verificación rápida del estado
python check_pipeline.py
```

### Paso 5 — Ver los logs

```powershell
# Log principal
type logs\pipeline.log

# Seguir en tiempo real (PowerShell)
Get-Content logs\pipeline.log -Wait
```

---

## 3. Subir archivos a la MV con FileZilla

### Configuración de la conexión en FileZilla

1. Abrir FileZilla → `Archivo > Gestor de sitios > Nuevo sitio`
2. Rellenar:

| Campo | Valor |
|---|---|
| Protocolo | SFTP - SSH File Transfer Protocol |
| Servidor | `hydra4-pre.unav.es` (pre) / `external.unav.es` (acceso externo) |
| Puerto | 22 |
| Modo de acceso | Normal |
| Usuario | `jvelazquezc` |
| Contraseña | (tu contraseña UNAV) |

3. Hacer clic en **Conectar**.

### Archivos que hay que subir

Navegar en el panel derecho (MV) hasta `~/UNAV/` y subir:

| Qué subir (panel izquierdo, local) | Dónde dejarlo (panel derecho, MV) |
|---|---|
| `linux_deploy/src/` completo | `~/UNAV/src/` |
| `models/modelo_final_grado.pkl` | `~/UNAV/models/` |
| `models/modelo_final_master.pkl` | `~/UNAV/models/` |
| `requirements.txt` | `~/UNAV/` |
| `check_pipeline.py` | `~/UNAV/` |
| `run_pipeline.sh` | `~/UNAV/` |
| `setup_cron.sh` | `~/UNAV/` |

> **No subir el `.env` por FileZilla.** Crearlo directamente en la MV (ver sección 5, paso 2) para evitar que quede en texto plano en el historial de transferencias.

---

## 4. Conectarse a la MV por SSH desde PowerShell

### Preproducción (red interna UNAV)

```powershell
ssh jvelazquezc@hydra4-pre.unav.es
```

### Preproducción (desde fuera de la red UNAV)

```powershell
ssh jvelazquezc@external.unav.es
```

Se pedirá la contraseña UNAV. Al conectar correctamente aparece el prompt del servidor Linux.

### Producción

```powershell
ssh probmatr@<host-mv-pro>
```

> Las credenciales y el host de la MV de producción los proporciona infraestructura.

---

## 5. Lanzar en preproducción

Una vez conectado por SSH:

### Paso 1 — Ir al directorio del proyecto

```bash
cd ~/UNAV
```

### Paso 2 — Crear o verificar el `.env` (solo la primera vez)

```bash
nano .env
```

Pegar el contenido con las credenciales de pre (ver sección 1). Guardar con `Ctrl+O` → `Enter` → `Ctrl+X`.

```bash
chmod 600 .env
```

### Paso 3 — Crear el entorno virtual (solo la primera vez)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
chmod +x run_pipeline.sh
```

### Paso 4 — Activar el entorno virtual (cada vez)

```bash
source .venv/bin/activate
```

### Paso 5 — Ejecutar la pipeline

```bash
# Pipeline completo
bash run_pipeline.sh

# Fases concretas
python src/pipeline.py --phases fase1
python src/pipeline.py --phases fase1 fase2
python src/pipeline.py --phases fase3 fase4

# Dry-run
python src/pipeline.py --dry-run
```

### Paso 6 — Ver logs en tiempo real

```bash
tail -f logs/pipeline.log
tail -f logs/cron.log
```

### Paso 7 — Instalar el cron (solo la primera vez)

```bash
bash setup_cron.sh
crontab -l    # verificar que aparece la entrada
```

La entrada instalada ejecuta el pipeline cada noche a las 03:00:
```
0 3 * * * ~/UNAV/run_pipeline.sh >> ~/UNAV/logs/cron.log 2>&1
```

---

## 6. Lanzar en producción

El proceso es idéntico a preproducción con estas diferencias:

| Diferencia | Pre | Pro |
|---|---|---|
| Usuario SSH | `jvelazquezc` | `probmatr` |
| Host MV | `hydra4-pre.unav.es` | A confirmar por infraestructura |
| `ORA_HOST` en `.env` | `racdb-pre.si.unav.es` | Host RAC producción |
| `ORA_SERVICE` en `.env` | `UNSIDPRE.UNAV` | Service name producción |
| `SF_URL` en `.env` | URL sandbox | URL org producción |
| Datos SF | Sandbox | Salesforce real |

Ver [DESPLIEGUE_PRODUCCION.md](DESPLIEGUE_PRODUCCION.md) para la guía completa paso a paso del primer despliegue en producción.

---

## 7. Verificar que la pipeline ha funcionado

### Verificación rápida

```bash
cd ~/UNAV && source .venv/bin/activate
python check_pipeline.py
```

### Verificar que PMAT_PREDICTION tiene datos de hoy

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

### Verificar que el write-back a Salesforce fue correcto

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
if err:
    print(df[df['STATUS']=='ERROR'][['OPP_ID','DETALLE']].head(10).to_string())
"
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

### Checklist diario rápido

```bash
# 1. Ver las últimas líneas del cron
tail -50 logs/cron.log

# 2. Ver si hay ERRORs en el log del pipeline
grep -i "error" logs/pipeline.log | tail -20

# 3. Verificación completa
python check_pipeline.py
```

---

## 8. Guía de errores frecuentes

### Errores de conexión Oracle

| Error | Causa más probable | Solución |
|---|---|---|
| `ORA-12541: TNS:no listener` | Host, puerto o service incorrecto | Verificar `ORA_HOST`, `ORA_PORT`, `ORA_SERVICE` en `.env`. Comprobar red: `nc -zv <host> 1521` |
| `ORA-01017: invalid username/password` | Credenciales incorrectas | Revisar `ORA_USER` y `ORA_PASSWORD` en `.env` |
| `ORA-00942: table or view does not exist` | `read_table` sin prefijo de esquema | Verificar que `oracle_connector.py` usa `schema.table` en los SELECT |
| `ORA-12899: value too large for column` | Campo DETALLE supera 100 caracteres | Verificar que `sf_writer.py` trunca DETALLE a 100 chars antes de insertar |
| `DPY-4027: no configuration directory` | DSN mal formado | Revisar `SF_URL` — debe incluir `https://` |
| `DPY-4018: cannot parse connect string` | Comentario inline en `.env` | Eliminar cualquier `# comentario` que esté en la misma línea que un valor |

**Diagnóstico rápido de conectividad Oracle:**
```bash
nc -zv racdb-pre.si.unav.es 1521
```
Si no responde, el problema es de red (VPN, firewall) o el host está mal escrito.

---

### Errores de conexión Salesforce

| Error | Causa más probable | Solución |
|---|---|---|
| `MissingSchema: Invalid URL` | `SF_URL` sin `https://` | Añadir `https://` al valor de `SF_URL` en `.env` |
| `INVALID_CLIENT_ID` / `INVALID_CLIENT_SECRET` | Credenciales de entorno incorrecto | Verificar que `SF_CLIENT_ID` y `SF_CLIENT_SECRET` corresponden al entorno (sandbox vs. pro) |
| `INSUFFICIENT_ACCESS_ON_CROSS_REFERENCE_ENTITY` | La Connected App no tiene permiso de escritura en Opportunity | Contactar con Usoa para revisar permisos de la Connected App |
| `FIELD_CUSTOM_VALIDATION_EXCEPTION` | Regla de validación en SF bloquea la escritura | Consultar con Usoa qué campos tienen validaciones activas |

---

### Errores de código Python

| Error | Causa más probable | Solución |
|---|---|---|
| `ModuleNotFoundError: oracle_connector` | Falta `sys.path.insert(0, 'src')` | Añadir `sys.path.insert(0, 'src')` antes de cualquier import local |
| `'DataFrame' has no attribute 'dtype'` | Columnas duplicadas en el DataFrame | Verificar que `preprocessor.py` deduplica columnas antes de iterar features |
| `KeyError: 'MODELO'` | Tabla `PMAT_PREDICTION` vacía o columna inexistente | Verificar que la fase3 completó sin errores antes de consultar la tabla |
| `FileNotFoundError: modelo_final_grado.pkl` | Modelos no copiados en `models/` | Copiar los `.pkl` en la carpeta `models/` del directorio activo |

---

### El cron no se ejecuta

```bash
# Verificar que el cron está instalado
crontab -l

# Ver si hay entradas de error en el log del sistema
grep CRON /var/log/syslog | tail -20

# Ejecutar manualmente para ver el error en tiempo real
cd ~/UNAV && source .venv/bin/activate && bash run_pipeline.sh
```

---

### La pipeline termina sin errores pero SF no se actualiza

1. Verificar que la fase4 se ejecutó: buscar `fase4` en `logs/pipeline.log`
2. Verificar el log de sync: los errores por oportunidad quedan en `PMAT_SF_SYNC_LOG`
3. Comprobar que `SF_PROB_FIELD` y `SF_CONF_FIELD` en el `.env` coinciden exactamente con los nombres de campo en Salesforce (sensible a mayúsculas)

---

## Contactos de soporte

| Rol | Para qué |
|---|---|
| Juan Velázquez | Dudas sobre el código, errores del pipeline, modelos |
| Mario Almendros | Operativa diaria, despliegues, seguimiento |
| Infraestructura UNAV | Credenciales Oracle, acceso MV, conectividad de red |
| Usoa (SF admin) | Credenciales Salesforce, permisos Connected App, campos SF |
