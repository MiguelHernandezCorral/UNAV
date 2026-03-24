"""
excel_loader.py
===============
Carga el dataset histórico del Excel DataSET_SF - V2.xlsx en Oracle,
creando tablas históricas (_HIST) para combinarlas con los datos actuales
de Salesforce (curso 2026/2027) en la fase de limpieza y modelado.

Tablas creadas en Oracle:
    OPPORTUNITY_HIST      ← hoja Oportunidad_OK     (70 297 filas)
    ACCOUNT_HIST          ← hoja Cuenta              (55 275 filas)
    ECBS_HIST             ← hoja ECB                 ( 6 282 filas)
    STAGE_HISTORY_HIST    ← hoja Historial_etapas_Oportunidad_OK (536 575 filas)
    ACTIVITY_HIST         ← hoja Historial_actividad_promocion    (62 170 filas)

Diferencias clave respecto a las tablas Oracle de Salesforce API:
  - Las columnas del Excel ya están en formato "notebook" (sin sufijo __C,
    nombres cortos, etc.) en lugar del formato completo de la API de SF.
  - El join Oportunidad×Cuenta usa ACCOUNTID ↔ ID18 (18 chars),
    no ACCOUNTID ↔ ID (15 chars) como en las tablas de la API.
  - YEARPERSONBIRTHDATE ya está calculado en Cuenta.
  - CH_ANTIGUO_ALUMNO__PC ya tiene el nombre correcto (con guión bajo).

Uso:
    python src/excel_loader.py
    python src/excel_loader.py --recreate    # elimina tablas HIST antes de insertar
    python src/excel_loader.py --excel-path "C:/ruta/al/archivo.xlsx"
"""

import sys
import logging
import logging.handlers
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from oracle_connector import OracleConnector

from __future__ import annotations

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
LOG_DIR  = BASE_DIR / "logs"
DOCS_DIR = BASE_DIR / "docs"
LOG_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

_root = logging.getLogger()
_root.setLevel(logging.INFO)
_fmt  = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s – %(message)s")

_console = logging.StreamHandler()
_console.setFormatter(_fmt)
_root.addHandler(_console)

_fh = logging.handlers.TimedRotatingFileHandler(
    LOG_DIR / "excel_loader.log", when="midnight", backupCount=7, encoding="utf-8"
)
_fh.setFormatter(_fmt)
_root.addHandler(_fh)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ruta por defecto del Excel histórico
# ---------------------------------------------------------------------------
EXCEL_DEFAULT = Path(r"C:/Users/jvelazquezc/Downloads/DataSET_SF - V2.xlsx")

# ---------------------------------------------------------------------------
# Mapeo hoja Excel → tabla Oracle + configuración de carga
# ---------------------------------------------------------------------------
# Formato: (nombre_hoja, tabla_oracle, pk_col_o_None)
#   pk_col = None  → insert simple (sin deduplicación)
#   pk_col = "ID"  → upsert via MERGE INTO con clave primaria dada
HOJAS_CONFIG: list[tuple[str, str, str | None]] = [
    ("Oportunidad_OK",                   "OPPORTUNITY_HIST",   "ID"),
    ("Cuenta",                           "ACCOUNT_HIST",       "ID18"),
    ("ECB",                              "ECBS_HIST",          "Id"),
    ("Historial_etapas_Oportunidad_OK",  "STAGE_HISTORY_HIST", None),
    ("Historial_actividad_promocion",    "ACTIVITY_HIST",      None),
]


# ---------------------------------------------------------------------------
# Limpieza mínima de columnas antes de insertar en Oracle
# ---------------------------------------------------------------------------

def _limpiar_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la limpieza mínima necesaria antes de insertar en Oracle:
    - Reemplaza NaN por None (Oracle NULL)
    - Convierte numpy booleans a Python bool
    - Trunca strings que excedan 4000 chars (límite CLOB/NVARCHAR2)
    """
    # numpy → Python nativos
    df = df.where(df.notna(), None)
    # Asegurar que las columnas de tipo object no contienen tipos numpy extraños
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(
            lambda x: str(x) if x is not None and not isinstance(x, str) else x
        )
    return df


# Tamaño de batch para inserts masivos (evita timeouts en Oracle con volúmenes grandes)
BATCH_SIZE = 5_000


# ---------------------------------------------------------------------------
# Carga principal
# ---------------------------------------------------------------------------

def _insert_batched(
    ora: OracleConnector,
    records: list[dict],
    tabla: str,
) -> dict:
    """
    Inserta records en Oracle en batches de BATCH_SIZE filas.
    Crea la tabla automáticamente si no existe (inferencia de tipos desde los primeros registros).
    Devuelve métricas: {total_excel, db_before, db_after, inserted}.
    """
    if not records:
        return {"total_excel": 0, "db_before": 0, "db_after": 0, "inserted": 0}

    # Inferencia de tipos con TODOS los registros (necesario para obtener la
    # anchura máxima real de cada columna y evitar ORA-12899).
    # La inferencia se hace en Python (sin tráfico Oracle), es rápida.
    ora.create_table_if_not_exists(records, tabla)
    db_before = ora.count_rows(tabla)

    total_insertados = 0
    n_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        n_batch = i // BATCH_SIZE + 1
        m = ora.insert_records(batch, tabla)
        total_insertados += m.get("inserted", 0)
        logger.info(
            "    Batch %d/%d (%d filas): %d insertadas acumuladas",
            n_batch, n_batches, len(batch), total_insertados,
        )

    db_after = ora.count_rows(tabla)
    return {
        "total_excel": len(records),
        "db_before":   db_before,
        "db_after":    db_after,
        "inserted":    db_after - db_before,
    }


def run(excel_path: Path = EXCEL_DEFAULT, recreate: bool = False) -> list[dict]:
    """
    Lee el Excel y carga las 5 hojas relevantes en Oracle usando INSERT simple
    con batches de BATCH_SIZE filas (evita timeouts con volúmenes grandes).
    Devuelve lista de métricas por tabla.

    Nota: Para tablas HIST siempre se usa INSERT (no MERGE INTO) porque la
    estrategia es recrear las tablas cada vez con --recreate.
    """
    start_ts = datetime.now(timezone.utc)
    resultados: list[dict] = []

    if not excel_path.exists():
        raise FileNotFoundError(f"Excel no encontrado: {excel_path}")

    logger.info("=" * 70)
    logger.info("CARGA HISTÓRICA – Excel → Oracle  [INICIO]")
    logger.info("=" * 70)
    logger.info("Fichero: %s", excel_path)

    # ── Leer todas las hojas relevantes de una vez ──────────────────────────
    logger.info("Leyendo hojas del Excel...")
    hojas_necesarias = [h for h, _, _ in HOJAS_CONFIG]
    dfs_excel: dict[str, pd.DataFrame] = pd.read_excel(
        excel_path, sheet_name=hojas_necesarias
    )
    logger.info("  Hojas leídas: %s", list(dfs_excel.keys()))

    with OracleConnector() as ora:
        for hoja, tabla, _pk_col in HOJAS_CONFIG:
            logger.info("─" * 60)
            logger.info("Hoja: %-40s → Oracle: %s", hoja, tabla)

            df = dfs_excel[hoja].copy()
            df = _limpiar_df(df)
            records = df.to_dict(orient="records")

            logger.info("  Filas Excel: %d | Columnas: %d", len(records), len(df.columns))

            if recreate:
                ora.drop_table_if_exists(tabla)

            # INSERT en batches (sin MERGE INTO para evitar timeouts con 70K filas)
            metrics = _insert_batched(ora, records, tabla)

            resultados.append({
                "hoja":       hoja,
                "tabla":      tabla,
                "excel":      metrics["total_excel"],
                "db_antes":   metrics["db_before"],
                "db_despues": metrics["db_after"],
                "insertados": metrics["inserted"],
            })
            logger.info(
                "  ✓ %s: %d filas Excel | DB antes: %d | DB después: %d | Insertados: %d",
                tabla,
                metrics["total_excel"],
                metrics["db_before"],
                metrics["db_after"],
                metrics["inserted"],
            )

    end_ts = datetime.now(timezone.utc)
    dur = (end_ts - start_ts).total_seconds()

    # ── Resumen ──────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("CARGA HISTÓRICA – Excel → Oracle  [FIN]  (%.1f s)", dur)
    logger.info("=" * 70)
    logger.info("")
    logger.info("%-40s %8s %8s %8s %10s",
                "Tabla", "Excel", "DB antes", "DB desp.", "Insertados")
    logger.info("-" * 80)
    for r in resultados:
        logger.info("%-40s %8d %8d %8d %10d",
                    r["tabla"], r["excel"], r["db_antes"],
                    r["db_despues"], r["insertados"])

    _write_docs(resultados, excel_path, recreate, start_ts, end_ts)
    return resultados


# ---------------------------------------------------------------------------
# Documentación
# ---------------------------------------------------------------------------

def _write_docs(
    resultados: list[dict],
    excel_path: Path,
    recreate: bool,
    start_ts: datetime,
    end_ts: datetime,
) -> Path:
    dur = (end_ts - start_ts).total_seconds()

    lines = [
        "# Carga Histórica – Excel → Oracle",
        "",
        "## ¿Qué hace este script?",
        "",
        "Lee el fichero histórico de Salesforce `DataSET_SF - V2.xlsx` y carga",
        "5 hojas en Oracle como tablas históricas (`*_HIST`). Estas tablas se",
        "combinan con los datos actuales de Salesforce (curso 2026/2027) en la",
        "fase de limpieza (`cleaner.py --include-historical`) para entrenar el",
        "modelo con datos históricos reales y predecir el curso vigente.",
        "",
        "## Diferencias entre tablas Oracle API y tablas HIST",
        "",
        "| Aspecto | Oracle API (SF 2026/27) | Tablas HIST (Excel histórico) |",
        "|---------|-------------------------|-------------------------------|",
        "| Nombres columnas | Formato API (`PL_CURSO_ACADEMICO__C`) | Formato notebook (`PL_CURSO_ACADEMICO`) |",
        "| Join Account | `ACCOUNTID → ID` (15 chars) | `ACCOUNTID → ID18` (18 chars) |",
        "| YEARPERSONBIRTHDATE | Se deriva de `PERSONBIRTHDATE` | Ya calculado en Excel |",
        "| CENTROENSENANZA | De OPPORTUNITY (relación anidada) | De ACCOUNT_HIST (columna directa) |",
        "| CH_ANTIGUO_ALUMNO__PC | Renombrado de `CH_ANTIGUOALUMNO__PC` | Nombre correcto directo |",
        "",
        "## Ejecución",
        "",
        f"- **Fichero fuente:** `{excel_path}`",
        f"- **Fecha/hora inicio:** {start_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Fecha/hora fin:**    {end_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Duración total:**    {dur:.1f} s",
        f"- **Modo:**              {'RECREAR tablas (--recreate)' if recreate else 'Insert acumulativo'}",
        "",
        "## Métricas por tabla",
        "",
        "| Tabla Oracle | Hoja Excel | Filas Excel | DB antes | DB después | Insertados |",
        "|---|---|---|---|---|---|",
    ]
    for r in resultados:
        lines.append(
            f"| `{r['tabla']}` | {r['hoja']} | {r['excel']:,} | {r['db_antes']:,} "
            f"| {r['db_despues']:,} | {r['insertados']:,} |"
        )

    lines += [
        "",
        "## Tablas creadas",
        "",
        "| Tabla | Descripción | PK |",
        "|---|---|---|",
        "| `OPPORTUNITY_HIST` | Oportunidades históricas (múltiples cursos) | `ID` (upsert) |",
        "| `ACCOUNT_HIST` | Cuentas/personas históricas | `ID18` (upsert) |",
        "| `ECBS_HIST` | Estudios coste y becas históricos | `Id` (upsert) |",
        "| `STAGE_HISTORY_HIST` | Historial de etapas histórico (536 K filas) | — (insert) |",
        "| `ACTIVITY_HIST` | Historial de actividades de promoción | — (insert) |",
        "",
        "## Uso en cleaner.py",
        "",
        "Una vez cargadas las tablas HIST, ejecutar:",
        "",
        "```bash",
        "python src/cleaner.py --recreate --include-historical",
        "```",
        "",
        "Esto combina:",
        "- Datos actuales de SF (2026/27): ~9 600 oportunidades × etapas",
        "- Datos históricos del Excel: ~70 000 oportunidades × etapas",
        "",
        f"- **Log completo:** `logs/excel_loader.log`",
        "",
        "---",
        f"*Generado automáticamente el {end_ts.strftime('%Y-%m-%d %H:%M')} UTC*",
    ]

    doc_path = DOCS_DIR / "EXCEL_LOADER.md"
    doc_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Documentación generada: %s", doc_path)
    return doc_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Carga el Excel histórico DataSET_SF - V2.xlsx en Oracle"
    )
    parser.add_argument(
        "--excel-path",
        type=Path,
        default=EXCEL_DEFAULT,
        help=f"Ruta al Excel. Por defecto: {EXCEL_DEFAULT}",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Elimina las tablas HIST en Oracle antes de insertar (esquema limpio)",
    )
    args = parser.parse_args()
    run(excel_path=args.excel_path, recreate=args.recreate)
