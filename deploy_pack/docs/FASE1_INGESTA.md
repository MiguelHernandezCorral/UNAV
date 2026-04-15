# Fase 1 – Ingesta Salesforce → Oracle

## ¿Qué hace esta fase?

Esta fase extrae todas las entidades necesarias de Salesforce mediante la
**REST API SOQL** (paginación automática de 2.000 registros por página),
aplana las relaciones anidadas (`RecordType.name`, `Campaign.Name`, etc.)
y las carga directamente en Oracle **sin generar CSVs intermedios**.

La carga se realiza mediante **MERGE INTO** (upsert):
- Si el registro **no existe** en Oracle → INSERT.
- Si el registro **ya existe** y algún campo ha cambiado → UPDATE.
- Si el registro ya existe y no ha cambiado → se ignora (sin escritura).

La inferencia de tipos Oracle (NUMBER, FLOAT, NVARCHAR2, CLOB) se realiza
automáticamente a partir de los valores Python nativos del JSON.
Las columnas CLOB se excluyen de la comparación de cambios (limitación Oracle).

## Conexión

| Parámetro | Valor |
|-----------|-------|
| Host      | `opportunity` ← variable `ORA_HOST` |
| Proxy     | `PMAT_USR[PMATOWNER]` |
| Esquema   | `PMATOWNER` |
| Modo      | oracledb thin (sin cliente Oracle) |

## Ejecución

- **Fecha/hora inicio:** 2026-03-26 09:45:01 UTC
- **Fecha/hora fin:**    2026-03-26 09:48:19 UTC
- **Duración total:**    198.5 s
- **Modo:**              Upsert incremental
- **Curso académico:**   2026/2027

## Resumen global

| Métrica | Valor |
|---------|-------|
| Tablas OK    | 10 / 10 |
| Tablas ERROR | 0 |
| Registros SF (total) | 235,269 |
| Registros en BD (total) | 235,269 |
| Insertados   | 0 |
| Actualizados (aprox.) | 235,269 |

## Detalle por entidad

| # | Entidad SF | Tabla Oracle | Registros SF | BD antes | BD después | Insertados | Actualizados | Estado |
|---|-----------|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `opportunity` | `OPPORTUNITY` | 9,605 | 9,605 | 9,605 | 0 | 9,605 | ✅ OK |
| 2 | `account` | `ACCOUNT` | 8,145 | 8,145 | 8,145 | 0 | 8,145 | ✅ OK |
| 3 | `ecbs` | `ECBS` | 4,126 | 4,126 | 4,126 | 0 | 4,126 | ✅ OK |
| 4 | `solban` | `SOLBAN` | 522 | 522 | 522 | 0 | 522 | ✅ OK |
| 5 | `cases` | `CASES` | 6,974 | 6,974 | 6,974 | 0 | 6,974 | ✅ OK |
| 6 | `email_results` | `EMAIL_RESULTS` | 4,770 | 4,770 | 4,770 | 0 | 4,770 | ✅ OK |
| 7 | `activity_history` | `ACTIVITY_HISTORY` | 550 | 550 | 550 | 0 | 550 | ✅ OK |
| 8 | `opp_field_history` | `OPP_FIELD_HISTORY` | 140,111 | 140,111 | 140,111 | 0 | 140,111 | ✅ OK |
| 9 | `pagos` | `PAGOS` | 7,166 | 7,166 | 7,166 | 0 | 7,166 | ✅ OK |
| 10 | `stage_history` | `STAGE_HISTORY` | 53,300 | 53,300 | 53,300 | 0 | 53,300 | ✅ OK |

## Notas técnicas

- **Paginación SF:** 2.000 registros por página (límite Salesforce Bulk).
- **Bind variables:** los nombres con caracteres especiales (`.`, `__r`) se
  sanitizan reemplazando caracteres no alfanuméricos por `_`.
- **CLOB en change_cond:** Oracle no permite comparar columnas CLOB con `!=`.
  Estas columnas se actualizan siempre que la fila haya coincidido.
- **Conteo de actualizados:** aproximado. Se calcula como
  `registros_SF – insertados`. Los registros coincidentes pero sin cambios
  no generan UPDATE y podrían inflar esta cifra.
- **Log completo:** `logs/sf_extract_all.log`

---
*Generado automáticamente el 2026-03-26 09:48 UTC*

---

*Autor: Viewnext (Juan Velázquez y Mario Almendros)*