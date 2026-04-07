# Pipeline UNAV — Paquete de despliegue a PRODUCCIÓN

**Fecha prevista de despliegue:** 13 de abril de 2026
**Preparado por:** Juan Velázquez
**Versión:** 1.0

---

## Qué contiene esta carpeta

Esta carpeta contiene todo lo necesario para instalar y ejecutar el pipeline de predicción de matrícula en la máquina de **producción**. Súbela completa a la nueva MV via FileZilla.

```
produccion_deploy/
├── src/                        ← Código fuente del pipeline
│   ├── pipeline.py             ← Orquestador principal (punto de entrada)
│   ├── sf_extractor.py         ← Fase 1: extrae datos de Salesforce → Oracle
│   ├── cleaner.py              ← Fase 2a: limpieza de datos
│   ├── preprocessor.py         ← Fase 2b: transformación y feature engineering
│   ├── predictor.py            ← Fase 3: carga modelos .pkl y genera predicciones
│   ├── sf_writer.py            ← Fase 4: envía probabilidades a Salesforce
│   ├── oracle_connector.py     ← Conexión y operaciones Oracle (UPSERT, MERGE)
│   ├── explainer.py            ← Generación de explicaciones SHAP (opcional)
│   ├── excel_loader.py         ← Carga de datos desde Excel (opcional)
│   └── sf_extract_all.py       ← Herramienta de extracción SF completa (diagnóstico)
├── docs/
│   ├── DESPLIEGUE_PRODUCCION.md ← LEER PRIMERO — guía completa de despliegue
│   ├── PIPELINE_DEV.md         ← Referencia técnica del pipeline
│   └── OPERACIONAL.md          ← Mantenimiento, cron, retención de logs
├── models/                     ← VACÍA — sube aquí los .pkl por FileZilla
├── logs/                       ← VACÍA — se puebla al ejecutar el pipeline
├── run_pipeline.sh             ← Script de ejecución principal
├── setup_cron.sh               ← Configura ejecución automática diaria (cron)
├── check_pipeline.py           ← Comprobación rápida del estado del pipeline
├── requirements.txt            ← Dependencias Python
└── .env.template               ← Plantilla de credenciales (renombrar a .env)
```

---

## Pasos rápidos (resumen)

> Para la guía detallada con resolución de problemas, ver `docs/DESPLIEGUE_PRODUCCION.md`

### 1. Antes de subir

- [ ] Obtener credenciales Oracle de producción (infraestructura)
- [ ] Obtener credenciales Salesforce de producción (Usoa)
- [ ] Tener los ficheros `models/*.pkl` listos para subir por separado

### 2. Subir a la MV

Via FileZilla:
- Esta carpeta completa → `/home/infra/jvelazquezc/UNAV/`
- Carpeta `models/` con los `.pkl` → `/home/infra/jvelazquezc/UNAV/models/`

### 3. En la MV — primer despliegue

```bash
ssh <usuario>@<nueva-mv-pro>
cd /home/infra/jvelazquezc/UNAV

# Crear .env con las credenciales reales
cp .env.template .env
nano .env          # rellenar los valores <...>
chmod 600 .env

# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
chmod +x run_pipeline.sh setup_cron.sh

# Verificar instalación
python -c "import sys; sys.path.insert(0,'src'); from pipeline import run_pipeline; print('OK')"

# Dry-run (sin escritura real)
python src/pipeline.py --dry-run

# Primera ejecución real
bash run_pipeline.sh

# Verificar resultados
python check_pipeline.py

# Configurar cron (ejecución diaria a las 06:00)
bash setup_cron.sh
```

### 4. Verificar en Salesforce

Tras la primera ejecución, confirmar con Usoa que los campos se han actualizado correctamente en los registros de oportunidades.

---

## Actualizaciones de código posteriores

Cuando haya un cambio en el código:

1. En Windows: ejecutar `sync_deploy.bat` (actualiza `linux_deploy/`)
2. Subir solo los `.py` modificados via FileZilla a la carpeta `src/`
3. No hace falta reinstalar el entorno salvo que cambie `requirements.txt`

---

## Campos que escribe el pipeline en Salesforce

| Campo SF | Descripción | Rango |
|---|---|---|
| `NU_Probabilidad_de_matricula__c` | Probabilidad de matrícula del modelo | 0–100 (entero) |
| `ProbabilityConfidence__c` | Confianza del modelo en la predicción | 0–100 (entero) |

**Solo se actualiza** un registro en Salesforce si su probabilidad ha cambiado respecto al último envío OK. Esto minimiza el consumo de la API.

---

## Contacto

Para dudas técnicas sobre el pipeline: Juan Velázquez
Para credenciales / permisos Salesforce: Usoa
Para acceso a la MV / Oracle: Infraestructura
