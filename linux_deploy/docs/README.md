# Pipeline UNAV — Despliegue en Linux

Carpeta lista para copiar a la máquina virtual Linux via FileZilla.
Ruta destino en la MV: `/home/infra/jvelazquezc/UNAV/`

---

## Primer despliegue (desde cero)

### 1. Subir archivos via FileZilla
Sube esta carpeta completa (`linux_deploy/`) a `/home/infra/jvelazquezc/UNAV/`.
Adicionalmente, sube la carpeta `models/` (ficheros `.pkl`) por separado — son grandes.

### 2. Crear entorno virtual
```bash
cd /home/infra/jvelazquezc/UNAV
python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Dar permisos al script
```bash
chmod +x run_pipeline.sh
```

### 4. Verificar que todo importa correctamente
```bash
python -c "
import sys; sys.path.insert(0, 'src')
from pipeline import run_pipeline, PHASE_REGISTRY
from oracle_connector import OracleConnector
from predictor import run_predictions_v2
print('OK —', sys.version)
"
```

### 5. Dry-run (sin escribir en Oracle/Salesforce)
```bash
python src/pipeline.py --dry-run
```
Resultado esperado: `OK: 4 | Errores: 0`

### 6. Probar conexión Oracle
```bash
nc -zv racdb-pre.si.unav.es 1521
python src/pipeline.py --phases fase2
```

### 7. Pipeline completo
```bash
bash run_pipeline.sh
```

---

## Actualizaciones posteriores

Cuando se haga un cambio en el código local:

1. Ejecutar `sync_deploy.bat` en Windows (en la raíz del proyecto)
2. Subir solo los `.py` modificados via FileZilla a `/home/infra/jvelazquezc/UNAV/src/`
3. No es necesario reinstalar el entorno virtual salvo que cambie `requirements.txt`

---

## Credenciales

Las credenciales están en `.env` (este fichero NO se sube a git).

| Variable | Valor |
|---|---|
| ORA_USER | PMAT_BTCH |
| ORA_SCHEMA | PMATOWNER |
| ORA_HOST | racdb-pre.si.unav.es |
| ORA_SERVICE | UNSIDPRE.UNAV |

El código construye automáticamente `PMAT_BTCH[PMATOWNER]` para la conexión proxy Oracle.

---

## Ejecución por fases

```bash
# Solo ingesta Salesforce → Oracle
bash run_pipeline.sh --phases fase1

# Solo limpieza
bash run_pipeline.sh --phases fase2

# Solo predicciones (si DATASET_LIMPIO ya existe en Oracle)
bash run_pipeline.sh --phases fase4

# Pipeline completo con historial de predicciones
bash run_pipeline.sh --save-hist

# Sin escribir en Oracle (prueba)
bash run_pipeline.sh --dry-run
```

## Logs

```
/home/infra/jvelazquezc/UNAV/logs/pipeline.log          # log principal
/home/infra/jvelazquezc/UNAV/logs/pipeline_YYYYMMDD_HHMMSS.log  # por ejecución
```
