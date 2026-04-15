# Onboarding — Guía para nuevos desarrolladores

Bienvenido al proyecto de predicción de matrícula de UNAV. Esta guía te pone en contexto y te da todo lo necesario para entender, ejecutar y modificar el pipeline.

---

## ¿Qué hace este proyecto?

Cada noche, un pipeline automático:
1. Extrae todas las oportunidades activas de Salesforce (candidatos a matricularse)
2. Limpia y transforma los datos
3. Aplica modelos de ML entrenados con el histórico de UNAV
4. Escribe en cada oportunidad de Salesforce la probabilidad de matrícula (0-100) y la confianza del modelo

El objetivo es que el equipo comercial de UNAV pueda priorizar su atención en los candidatos con mayor probabilidad de matricularse.

---

## Estructura del repositorio

```
UNAV/
├── src/                    # Código fuente del pipeline
│   ├── pipeline.py         # Orquestador — punto de entrada
│   ├── sf_extract_all.py   # Fase 1: extrae SF → Oracle
│   ├── sf_extractor.py     # Cliente Salesforce REST API
│   ├── cleaner.py          # Fase 2: limpieza → DATASET_LIMPIO
│   ├── preprocessor.py     # Preprocesado pre-PyCaret
│   ├── predictor.py        # Fase 3: predicciones → PMAT_PREDICTION
│   ├── explainer.py        # Explicaciones SHAP por fila
│   ├── sf_writer.py        # Fase 4: write-back → Salesforce
│   ├── oracle_connector.py # Cliente Oracle (MERGE, INSERT, SELECT)
│   ├── excel_loader.py     # Carga histórico Excel (uso puntual)
│   └── test_*.py           # Tests unitarios
├── models/                 # Modelos .pkl (NO en git — transferir manualmente)
│   ├── modelo_grado.pkl
│   └── modelo_master.pkl
├── docs/                   # Documentación
│   ├── ONBOARDING.md       # Este archivo
│   ├── ARQUITECTURA.md     # Diagrama y descripción de la arquitectura
│   ├── CODIGO_PIPELINE.md  # Documentación detallada de cada módulo
│   ├── EJECUCION.md        # Cómo lanzar el pipeline en pre y pro
│   ├── OPERACIONAL.md      # Operativa diaria, cron, monitoreo
│   └── DESPLIEGUE_PRODUCCION.md  # Guía de despliegue completa
├── notebooks/              # Análisis y modelado exploratorio
│   ├── 01_Limpieza.ipynb   # Origen de la lógica de cleaner.py
│   ├── 03_Modelado.ipynb   # Origen de la lógica de preprocessor.py
│   └── ...
├── produccion_deploy/      # Copia de src/ lista para subir a producción
├── linux_deploy/           # Copia de src/ para la MV Linux de pre
├── check_pipeline.py       # Script de verificación rápida
├── run_pipeline.sh         # Lanzador bash del pipeline
├── setup_cron.sh           # Instala el cron diario
└── requirements.txt        # Dependencias Python
```

---

## Configuración del entorno local

### 1 — Clonar el repositorio

```bash
git clone https://github.com/MiguelHernandezCorral/UNAV.git
cd UNAV
```

### 2 — Crear entorno virtual

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3 — Crear el .env

Copia el template y rellena con las credenciales de preproducción:

```bash
cp produccion_deploy/.env.template .env
```

Edita `.env`:
```dotenv
ORA_HOST=racdb-pre.si.unav.es
ORA_PORT=1521
ORA_SERVICE=UNSIDPRE.UNAV
ORA_USER=PMAT_BTCH
ORA_SCHEMA=PMATOWNER
ORA_PASSWORD=<pedir a Juan>

SF_URL=https://unav--fulladm.sandbox.my.salesforce.com
SF_SITE=
SF_CLIENT_ID=<pedir a Juan>
SF_CLIENT_SECRET=<pedir a Juan>
SF_API_VERSION=60.0
SF_PROB_FIELD=NU_Probabilidad_de_matricula__c
SF_CONF_FIELD=ProbabilityConfidence__c
```

> Las credenciales de producción las proporciona infraestructura (Oracle) y Usoa (Salesforce).

### 4 — Obtener los modelos

Los modelos `.pkl` no están en git. Pedirlos a Juan y copiarlos en `models/`.

### 5 — Verificar instalación

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from pipeline import run_pipeline, PHASE_REGISTRY
from oracle_connector import OracleConnector
from predictor import run_predictions_v2
from sf_writer import run as sf_run
print('OK —', sys.version)
"
```

---

## Ejecutar el pipeline localmente

```bash
# Dry-run — simula todo sin escribir datos
python src/pipeline.py --dry-run

# Solo fase1 (ingesta SF → Oracle)
python src/pipeline.py --phases fase1

# Pipeline completo
python src/pipeline.py
```

---

## Ejecutar los tests

```bash
python -m pytest src/test_*.py -v
```

---

## Flujo de desarrollo

### Ramas
- `main` — código estable, lo que está o va a ir a producción
- `ramaJuan` — desarrollo Juan Velázquez
- `ramaMario` — desarrollo Mario Almendros

### Sincronización con producción

Cada cambio en `src/` debe sincronizarse manualmente con `produccion_deploy/src/` y `linux_deploy/src/` antes de subir a la MV. Los tres directorios deben ser idénticos en producción.

### Desplegar un cambio en la MV

1. Modifica el archivo en `src/`
2. Copia a `produccion_deploy/src/` y `linux_deploy/src/`
3. Commit y push a main
4. Sube el archivo modificado a la MV vía FileZilla

---

## Conceptos clave que debes entender

### Conexión Oracle proxy
El pipeline conecta como `PMAT_BTCH[PMATOWNER]` — el usuario `PMAT_BTCH` actúa en nombre del esquema `PMATOWNER`. Por eso todas las tablas se crean y leen con el prefijo `PMATOWNER.NOMBRE_TABLA`.

### MERGE INTO (upsert)
`oracle_connector.py` no hace INSERT simple sino MERGE INTO — si el registro ya existe lo actualiza, si no existe lo inserta. La clave de merge se infiere del nombre de la tabla (ej: OPP_ID para OPPORTUNITY).

### Segmentación grado/master
El predictor entrena y predice por separado para Grado y Máster. Hay un modelo `.pkl` por cada tipo. El preprocessor aplica transformaciones específicas según el tipo.

### Vista PMAT_PRED_ACTUAL
La fase3 crea/actualiza automáticamente una vista Oracle que agrupa `PMAT_PREDICTION` por OPP_ID devolviendo solo la predicción más reciente por oportunidad. La fase4 lee de esta vista.

---

## Personas del proyecto

| Rol | Nombre | Contacto |
|---|---|---|
| Responsable técnico | Juan Velázquez | — |
| Desarrollador | Mario Almendros | — |
| Cliente SF admin | Usoa | Credenciales SF, permisos, campos |
| Infraestructura | — | Credenciales Oracle, acceso MV |

---

## Documentación adicional

- [ARQUITECTURA.md](ARQUITECTURA.md) — diagrama completo del flujo de datos
- [CODIGO_PIPELINE.md](CODIGO_PIPELINE.md) — documentación detallada de cada módulo
- [EJECUCION.md](EJECUCION.md) — comandos para lanzar en pre y pro
- [OPERACIONAL.md](OPERACIONAL.md) — monitoreo, cron, purga de logs
- [DESPLIEGUE_PRODUCCION.md](DESPLIEGUE_PRODUCCION.md) — guía de despliegue completa
