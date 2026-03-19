"""
oracle_connector.py
===================
Gestiona la conexión a Oracle y la carga de datos directamente desde
los registros JSON de Salesforce (list[dict] aplanados).

Operaciones:
  - Conexión proxy  pmat_usr[pmatowner]
  - CREATE TABLE automático inferido desde los valores JSON
  - DROP TABLE (para recrear con esquema actualizado)
  - Upsert genérico via MERGE INTO  →  devuelve métricas detalladas
  - Insert histórico (para tabla PREDICCIONES_PROBABILIDAD_MATRICULA)
  - Lectura de tabla a list[dict]
  - COUNT de filas

Variables de entorno (.env):
    ORA_HOST, ORA_PORT, ORA_SERVICE, ORA_USER, ORA_SCHEMA, ORA_PASSWORD
"""

import os
import re
import logging
import oracledb
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inferencia de tipos Oracle desde valores Python nativos (JSON)
# ---------------------------------------------------------------------------

def _infer_ora_type(values: list) -> str:
    """
    Infiere el tipo Oracle a partir de una lista de valores Python nativos
    procedentes del JSON de Salesforce.
    Ignora None al inferir; si todos son None devuelve NVARCHAR2(100).
    """
    non_null = [v for v in values if v is not None]
    if not non_null:
        return "NVARCHAR2(100)"

    # bool debe ir antes de int (bool es subclase de int en Python)
    if all(isinstance(v, bool) for v in non_null):
        return "NUMBER(1)"
    if all(isinstance(v, int) for v in non_null):
        return "NUMBER(19)"
    if all(isinstance(v, float) for v in non_null):
        return "FLOAT"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
        return "FLOAT"

    # Strings: determinar longitud máxima
    str_vals = [str(v) for v in non_null]
    max_len = max(len(s) for s in str_vals)
    if max_len <= 100:
        return "NVARCHAR2(100)"
    if max_len <= 500:
        return "NVARCHAR2(500)"
    if max_len <= 2000:
        return "NVARCHAR2(2000)"
    return "CLOB"


def _ora_type_to_db_type(ora_type: str):
    """Mapea un tipo Oracle DDL al DB_TYPE de oracledb para setinputsizes."""
    if any(t in ora_type for t in ("NUMBER", "FLOAT", "INTEGER")):
        return oracledb.DB_TYPE_NUMBER
    if "TIMESTAMP" in ora_type:
        return oracledb.DB_TYPE_TIMESTAMP
    if "DATE" in ora_type:
        return oracledb.DB_TYPE_DATE
    if "CLOB" in ora_type:
        return oracledb.DB_TYPE_CLOB
    return oracledb.DB_TYPE_NVARCHAR


def _bind(col: str) -> str:
    """Convierte un nombre de columna a bind variable válida (sin puntos ni chars especiales)."""
    return re.sub(r"[^A-Za-z0-9_]", "_", col)


def _coerce_value(value, ora_type: str):
    """
    Convierte un valor Python nativo al tipo correcto para Oracle.
    - bool     → int para NUMBER, '1'/'0' para string
    - int/float → nativo para NUMBER, str() para NVARCHAR/CLOB
    - str      → float/None para NUMBER (si no numérico → None)
    - None     → None

    Garantiza que oracledb recibe siempre el Python type esperado
    según el DB_TYPE declarado en setinputsizes.
    """
    if value is None:
        return None
    if "NUMBER" in ora_type or "FLOAT" in ora_type:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return value
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    # Columna de tipo string (NVARCHAR2, CLOB): garantizar str Python
    if isinstance(value, bool):
        return "1" if value else "0"
    if not isinstance(value, str):
        return str(value)
    return value


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class OracleConnector:
    """
    Cliente Oracle para el pipeline UNAV.
    Trabaja directamente con list[dict] del JSON de Salesforce.

    Uso:
        with OracleConnector() as ora:
            metrics = ora.upsert_records(records, "OPPORTUNITY", pk_col="Id")
    """

    def __init__(self):
        self.host     = os.getenv("ORA_HOST", "")
        self.port     = int(os.getenv("ORA_PORT", 1521))
        self.service  = os.getenv("ORA_SERVICE", "")
        self.user     = os.getenv("ORA_USER", "")
        self.schema   = os.getenv("ORA_SCHEMA", self.user)
        self.password = os.getenv("ORA_PASSWORD", "")
        self._conn: oracledb.Connection | None = None
        self._validate_config()

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        missing = [k for k, v in {
            "ORA_HOST": self.host, "ORA_SERVICE": self.service,
            "ORA_USER": self.user, "ORA_PASSWORD": self.password,
        }.items() if not v]
        if missing:
            raise EnvironmentError(f"Faltan variables de entorno de Oracle: {', '.join(missing)}")

    # ------------------------------------------------------------------
    # Conexión
    # ------------------------------------------------------------------

    def connect(self) -> None:
        dsn = oracledb.makedsn(self.host, self.port, service_name=self.service)
        proxy_user = f"{self.user}[{self.schema}]" if self.schema.upper() != self.user.upper() else self.user
        self._conn = oracledb.connect(user=proxy_user, password=self.password, dsn=dsn)
        logger.info("Conexión Oracle establecida: %s@%s/%s", proxy_user, self.host, self.service)

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Conexión Oracle cerrada.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    @property
    def conn(self) -> oracledb.Connection:
        if not self._conn:
            self.connect()
        return self._conn

    # ------------------------------------------------------------------
    # Inferencia de esquema desde registros JSON
    # ------------------------------------------------------------------

    def _infer_schema(self, records: list[dict]) -> dict[str, str]:
        """
        Infiere {COLUMN_NAME: ORA_TYPE} a partir de todos los registros.
        Recorre todos los registros para encontrar el tipo más representativo.
        """
        col_values: dict[str, list] = {}
        for rec in records:
            for k, v in rec.items():
                col_values.setdefault(k.upper(), []).append(v)
        return {col: _infer_ora_type(vals) for col, vals in col_values.items()}

    # ------------------------------------------------------------------
    # DDL
    # ------------------------------------------------------------------

    def _table_exists(self, table_name: str) -> bool:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM all_tables WHERE owner=:s AND table_name=:t",
            s=self.schema.upper(), t=table_name.upper(),
        )
        exists = cur.fetchone()[0] > 0
        cur.close()
        return exists

    def count_rows(self, table_name: str) -> int:
        """Cuenta las filas actuales de una tabla Oracle."""
        cur = self.conn.cursor()
        cur.execute(f'SELECT COUNT(*) FROM "{table_name.upper()}"')
        n = cur.fetchone()[0]
        cur.close()
        return n

    def drop_table_if_exists(self, table_name: str) -> None:
        """DROP TABLE si existe. Usado para recrear tabla con esquema actualizado."""
        table_upper = table_name.upper()
        if self._table_exists(table_upper):
            cur = self.conn.cursor()
            cur.execute(f'DROP TABLE "{table_upper}"')
            self.conn.commit()
            cur.close()
            logger.info("Tabla %s.%s eliminada (recreación).", self.schema, table_upper)

    def create_table_if_not_exists(
        self, records: list[dict], table_name: str, pk_col: str | None = None
    ) -> None:
        """Crea la tabla en Oracle si no existe, inferiendo tipos desde los registros."""
        table_upper = table_name.upper()
        if self._table_exists(table_upper):
            logger.info("Tabla %s.%s ya existe.", self.schema, table_upper)
            return

        schema = self._infer_schema(records)
        col_defs = [f'    "{col}" {otype}' for col, otype in schema.items()]
        if pk_col:
            pk_upper = pk_col.upper()
            col_defs.append(f'    CONSTRAINT PK_{table_upper} PRIMARY KEY ("{pk_upper}")')

        ddl = f'CREATE TABLE "{table_upper}" (\n' + ",\n".join(col_defs) + "\n)"
        cur = self.conn.cursor()
        logger.info("Creando tabla %s.%s ...", self.schema, table_upper)
        cur.execute(ddl)
        self.conn.commit()
        cur.close()
        logger.info("Tabla %s.%s creada.", self.schema, table_upper)

    # ------------------------------------------------------------------
    # Upsert genérico via MERGE INTO
    # ------------------------------------------------------------------

    def upsert_records(
        self,
        records: list[dict],
        table_name: str,
        pk_col: str,
    ) -> dict:
        """
        Carga registros JSON aplanados en Oracle via MERGE INTO.
        Crea la tabla automáticamente si no existe.
        Solo actualiza filas donde al menos un campo haya cambiado
        (columnas CLOB excluidas de la comparación).

        Args:
            records:    list[dict] con los registros aplanados de Salesforce
            table_name: nombre de la tabla Oracle
            pk_col:     nombre del campo clave primaria (sensible a mayúsculas del JSON)

        Returns:
            dict con métricas:
              sf_count   – registros recibidos de Salesforce
              db_before  – filas en Oracle antes del upsert
              db_after   – filas en Oracle después del upsert
              inserted   – filas nuevas insertadas (db_after - db_before)
              updated    – filas actualizadas (aproximado: affected - inserted)
        """
        if not records:
            logger.warning("Sin registros para %s, se omite upsert.", table_name)
            return {"sf_count": 0, "db_before": 0, "db_after": 0, "inserted": 0, "updated": 0}

        self.create_table_if_not_exists(records, table_name, pk_col)

        table_upper = table_name.upper()
        pk_upper    = pk_col.upper()
        schema      = self._infer_schema(records)
        cols        = list(schema.keys())
        non_pk_cols = [c for c in cols if c != pk_upper]

        # Columnas CLOB: no se pueden comparar con != en Oracle
        # Se incluyen en UPDATE SET pero se excluyen del change_cond
        clob_cols       = {c for c in non_pk_cols if "CLOB" in schema[c]}
        comparable_cols = [c for c in non_pk_cols if c not in clob_cols]

        # bind names: sin caracteres especiales
        bnd = {c: _bind(c) for c in cols}

        # Las columnas de la subconsulta DUAL necesitan alias para que
        # src."COL" sea válido en el ON/SET/VALUES del MERGE.
        src_cols    = ", ".join(f':{bnd[c]} AS "{c}"' for c in cols)
        match_cond  = f'tgt."{pk_upper}" = src."{pk_upper}"'
        update_set  = ", ".join(f'tgt."{c}" = src."{c}"' for c in non_pk_cols)
        insert_cols = ", ".join(f'"{c}"' for c in cols)
        insert_vals = ", ".join(f'src."{c}"' for c in cols)

        # change_cond: solo columnas comparables (no CLOB, no PK)
        # "1=1" cuando solo hay CLOBs → siempre actualizar filas coincidentes
        if comparable_cols:
            change_cond = " OR ".join(
                f'(tgt."{c}" != src."{c}" OR '
                f'(tgt."{c}" IS NULL AND src."{c}" IS NOT NULL) OR '
                f'(tgt."{c}" IS NOT NULL AND src."{c}" IS NULL))'
                for c in comparable_cols
            )
        elif non_pk_cols:
            change_cond = "1=1"   # solo CLOBs → actualizar siempre
        else:
            change_cond = "1=0"   # sin columnas no-PK → nunca actualizar

        merge_sql = f"""
            MERGE INTO "{table_upper}" tgt
            USING (SELECT {src_cols} FROM DUAL) src
            ON ({match_cond})
            WHEN MATCHED THEN
                UPDATE SET {update_set}
                WHERE {change_cond}
            WHEN NOT MATCHED THEN
                INSERT ({insert_cols})
                VALUES ({insert_vals})
        """

        # Preparar filas con bind names y tipos coercionados
        rows = [
            {bnd[c]: _coerce_value(rec.get(c.lower()) or rec.get(c), schema[c]) for c in cols}
            for rec in [{k.upper(): v for k, v in r.items()} for r in records]
        ]

        # setinputsizes: evita ambigüedad de tipo cuando el primer valor es None
        bind_sizes = {bnd[c]: _ora_type_to_db_type(ot) for c, ot in schema.items()}

        # ── Métricas ──────────────────────────────────────────────────
        db_before = self.count_rows(table_upper)

        cursor = self.conn.cursor()
        cursor.setinputsizes(**bind_sizes)
        cursor.executemany(merge_sql, rows)
        self.conn.commit()
        cursor.close()

        db_after = self.count_rows(table_upper)
        inserted = db_after - db_before
        updated  = len(rows) - inserted   # aproximado: no cuenta filas sin cambios

        metrics = {
            "sf_count":  len(rows),
            "db_before": db_before,
            "db_after":  db_after,
            "inserted":  inserted,
            "updated":   updated,
        }
        logger.info(
            "Upsert %s — SF: %d | DB antes: %d | insertadas: %d | actualizadas: ~%d",
            table_upper, metrics["sf_count"], metrics["db_before"],
            metrics["inserted"], metrics["updated"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Insert histórico
    # ------------------------------------------------------------------

    def insert_records(self, records: list[dict], table_name: str) -> dict:
        """
        Insert simple sin deduplicación.
        Usado para el histórico de PREDICCIONES_PROBABILIDAD_MATRICULA.

        Returns:
            dict con métricas: sf_count, db_before, db_after, inserted.
        """
        if not records:
            logger.warning("Sin registros para %s, se omite insert.", table_name)
            return {"sf_count": 0, "db_before": 0, "db_after": 0, "inserted": 0}

        self.create_table_if_not_exists(records, table_name)

        table_upper = table_name.upper()
        schema      = self._infer_schema(records)
        cols        = list(schema.keys())
        bnd         = {c: _bind(c) for c in cols}

        insert_cols = ", ".join(f'"{c}"' for c in cols)
        insert_vals = ", ".join(f':{bnd[c]}' for c in cols)
        sql = f'INSERT INTO "{table_upper}" ({insert_cols}) VALUES ({insert_vals})'

        rows = [
            {bnd[c]: _coerce_value(rec.get(c.lower()) or rec.get(c), schema[c]) for c in cols}
            for rec in [{k.upper(): v for k, v in r.items()} for r in records]
        ]

        bind_sizes = {bnd[c]: _ora_type_to_db_type(ot) for c, ot in schema.items()}

        db_before = self.count_rows(table_upper)

        cursor = self.conn.cursor()
        cursor.setinputsizes(**bind_sizes)
        cursor.executemany(sql, rows)
        self.conn.commit()
        cursor.close()

        db_after = self.count_rows(table_upper)
        metrics = {
            "sf_count":  len(rows),
            "db_before": db_before,
            "db_after":  db_after,
            "inserted":  db_after - db_before,
        }
        logger.info(
            "Insert %s — SF: %d | DB antes: %d | insertadas: %d",
            table_upper, metrics["sf_count"], metrics["db_before"], metrics["inserted"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Lectura
    # ------------------------------------------------------------------

    def read_table(self, table_name: str) -> list[dict]:
        """Lee una tabla Oracle completa y la devuelve como list[dict]."""
        table_upper = table_name.upper()
        cur = self.conn.cursor()
        cur.execute(f'SELECT * FROM "{table_upper}"')
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        cur.close()
        logger.info("Leída tabla %s: %d filas x %d columnas.", table_upper, len(rows), len(cols))
        return rows
