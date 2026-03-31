# Instrucciones para Claude Code

## Reglas generales

- No revises documentación a menos que se te pida expresamente.
- Siempre que hagas una modificación en el repositorio, aplica el mismo cambio en `linux_deploy/`:
  - `src/*.py` → `linux_deploy/src/`
  - `docs/*.md` → `linux_deploy/docs/`
  - `run_pipeline.sh` → `linux_deploy/run_pipeline.sh`
  - `requirements.txt` → `linux_deploy/requirements.txt`
  - Scripts raíz (`*.py`, `*.sh`) → `linux_deploy/` si procede
  - **Excepción:** `linux_deploy/.env` nunca se sobreescribe.
- Cuando des comandos para ejecutar en la MV de Linux, incluye siempre primero:
  ```bash
  ssh jvelazquezc@hydra4-pre.unav.es
  cd /home/infra/jvelazquezc/UNAV
  source .venv/bin/activate
  ```
