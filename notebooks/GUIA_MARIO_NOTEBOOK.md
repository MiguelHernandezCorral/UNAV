# Guía paso a paso para ejecutar el notebook de revisión (Mario)

> Guía para `2026-3-24_Plan de revision.ipynb`
> Preparada por Juan — para ejecutar desde VS Code con Jupyter

---

## ANTES DE EMPEZAR — Problemas que tiene el notebook actual

El notebook tiene **3 problemas principales** que hay que corregir antes de ejecutarlo:

| # | Problema | Síntoma |
|---|----------|---------|
| 1 | DSN de Oracle mal formateado | `DPY-4027: no configuration directory` |
| 2 | Credenciales escritas directamente en el código | No se leerán desde `.env` |
| 3 | Ruta de `src/` no apunta al lugar correcto | `ModuleNotFoundError: oracle_connector` |

---

## PASO 1 — Verificar que tienes Python y las librerías instaladas

Abre una **terminal** en VS Code (`Ctrl + ñ` o `Terminal → New Terminal`) y ejecuta:

```bash
python --version
```

Debe decir `Python 3.13.x`. Si no aparece, instala Python desde [python.org](https://python.org).

Luego instala las librerías necesarias:

```bash
pip install oracledb pandas python-dotenv matplotlib
```

> Si ya las tienes instaladas, pip dirá "already satisfied" — sin problema.

---

## PASO 2 — Configurar el archivo `.env`

El notebook necesita leer las credenciales de Oracle desde un archivo `.env`.
Este archivo **ya existe** en la raíz del proyecto (`UNAV/.env`).

Ábrelo y verifica que contiene exactamente esto (con el `//` al inicio del DSN):

```
DB_USER=PMAT_BTCH[PMATOWNER]
DB_PASS=w6IT%_)M>&
DB_DSN=//racdb-pre.si.unav.es:1521/UNSIDPRE.UNAV
```

> **IMPORTANTE — dos reglas para el `.env`:**
> 1. El `//` al inicio del DSN es obligatorio. Sin él, Oracle no sabe que es una dirección de red y da error `DPY-4027`.
> 2. **No pongas comentarios al final de una línea de valor** (p.ej. `DB_DSN=//host... # esto es un comentario`). `python-dotenv` los incluye como parte del valor y Oracle da error `DPY-4018`. Los comentarios solo funcionan en línea propia empezando por `#`.

---

## PASO 3 — Seleccionar el kernel correcto en VS Code

Cuando abras el notebook en VS Code:

1. Arriba a la derecha verás un botón que dice **"Select Kernel"** (o el nombre de un Python)
2. Haz clic → selecciona **"Python 3.13 (Global)"** o el entorno donde instalaste las librerías
3. Si no aparece ningún Python, reinicia VS Code y vuelve a abrir el notebook

> Si ves un kernel con nombre raro tipo `ipykernel_...` o `base (conda)`, **no lo uses** — esos pueden no tener oracledb instalado.

---

## PASO 4 — Corregir las celdas del notebook

### Celda 1 (Imports y conexión) — REEMPLAZAR el bloque de credenciales

Busca en el notebook estas líneas:

```python
# MAL — credenciales hardcodeadas
DB_USER = "PMAT_BTCH[PMATOWNER]"
DB_PASS = "w6IT%_)M>&"
DB_DSN  = "racdb-pre.si.unav.es:1521/UNSIDPRE.UNAV"
```

Y sustitúyelas por:

```python
# BIEN — lee desde .env
import os
from dotenv import load_dotenv

load_dotenv()  # Carga el archivo .env de la raíz del proyecto

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_DSN  = os.getenv("DB_DSN")   # Debe ser //host:puerto/servicio

# Verificación rápida (no imprime la contraseña)
print("Usuario:", DB_USER)
print("DSN:", DB_DSN)
print("Contraseña cargada:", "Sí" if DB_PASS else "NO — revisa el .env")
```

### Celda de conexión Oracle — VERIFICAR el DSN

```python
import oracledb

conn = oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN)
print("Conexión OK")
```

Si da error `DPY-4027`, es porque el `.env` tiene el DSN sin `//`. Vuelve al Paso 2.

### Celda con sys.path — CORREGIR el orden de los imports

**Este es el error más frecuente.** El `sys.path` debe ir **antes** de cualquier `import` local. Si pones el `from oracle_connector import ...` antes de añadir `src/` a la ruta, Python falla porque todavía no sabe dónde buscar.

MAL (el orden que tiene el notebook):
```python
from oracle_connector import OracleConnector   # ← FALLA: src/ todavía no está en la ruta
import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))  # ← demasiado tarde
```

BIEN (orden correcto):
```python
import sys, os
from dotenv import load_dotenv
load_dotenv()

# PRIMERO añadir la ruta, LUEGO importar módulos locales
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))
print("Ruta src:", os.path.join(project_root, 'src'))

import oracledb
import pandas as pd
from oracle_connector import OracleConnector   # ← ahora sí funciona
```

---

## PASO 5 — Ejecutar el notebook celda a celda

**No pulses "Run All" de golpe.** Ejecuta celda a celda con `Shift + Enter` y comprueba que cada una termina sin error antes de seguir.

Orden correcto:

1. Celda de instalación (`%pip install ...`) — espera a que termine
2. Celda de imports y carga del `.env`
3. Celda de conexión Oracle — **aquí es donde más fallos ocurren**
4. Celda de carga de datos (`pd.read_sql`)
5. El resto de análisis

---

## ERRORES COMUNES Y SOLUCIONES

### Error: `DPY-4027: no configuration directory`
**Causa:** El DSN no tiene `//` al inicio
**Solución:** En `.env` pon `DB_DSN=//racdb-pre.si.unav.es:1521/UNSIDPRE.UNAV`

---

### Error: `DPY-4018: cannot parse connect string`
**Causa:** El DSN tiene un comentario en la misma línea (`DB_DSN=//host... # comentario`). `python-dotenv` incluye el comentario como parte del valor.
**Solución:** Elimina el comentario de esa línea. Debe quedar solo:
```
DB_DSN=//racdb-pre.si.unav.es:1521/UNSIDPRE.UNAV
```

---

### Error: `ModuleNotFoundError: No module named 'oracledb'`
**Causa:** La librería no está instalada en el kernel seleccionado
**Solución:** En una celda del notebook, ejecuta:
```python
%pip install oracledb
```
Luego **reinicia el kernel** (botón de reinicio arriba) y vuelve a ejecutar desde el inicio.

---

### Error: `ModuleNotFoundError: No module named 'oracle_connector'`
**Causa:** El `from oracle_connector import ...` aparece antes del `sys.path.insert(...)` en el código
**Solución:** Reordena la celda: primero el bloque `sys.path`, luego los imports locales. Ver el ejemplo en el Paso 4.

---

### Warning: `UserWarning: pandas only supports SQLAlchemy connectable`
**Causa:** pandas avisa que prefiere SQLAlchemy sobre conexiones directas
**Efecto:** El código funciona igual, es solo un aviso
**Para eliminarlo** (opcional):
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
```

---

### Error: `DatabaseError: ORA-01017: invalid username/password`
**Causa:** Las credenciales en `.env` son incorrectas
**Solución:** Verifica con Juan o con el equipo que las credenciales del `.env` son correctas

---

## RESUMEN RÁPIDO (checklist)

- [ ] `pip install oracledb pandas python-dotenv matplotlib` ejecutado
- [ ] Archivo `.env` tiene `DB_DSN=//...` (con `//` al inicio)
- [ ] Kernel seleccionado es Python 3.13 con las librerías instaladas
- [ ] Credenciales leídas con `load_dotenv()`, no hardcodeadas
- [ ] `sys.path` apunta correctamente a `src/`
- [ ] Ejecutado celda a celda, no "Run All"
