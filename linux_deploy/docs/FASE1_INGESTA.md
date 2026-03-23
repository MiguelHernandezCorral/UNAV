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

- **Fecha/hora inicio:** 2026-03-19 09:41:57 UTC
- **Fecha/hora fin:**    2026-03-19 09:46:43 UTC
- **Duración total:**    286.5 s
- **Modo:**              RECREAR tablas (--recreate)
- **Curso académico:**   2026/2027

## Resumen global

| Métrica | Valor |
|---------|-------|
| Tablas OK    | 10 / 10 |
| Tablas ERROR | 0 |
| Registros SF (total) | 235,217 |
| Registros en BD (total) | 235,217 |
| Insertados   | 235,217 |
| Actualizados (aprox.) | 0 |

## Detalle por entidad

| # | Entidad SF | Tabla Oracle | Registros SF | BD antes | BD después | Insertados | Actualizados | Estado |
|---|-----------|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `opportunity` | `OPPORTUNITY` | 9,603 | 0 | 9,603 | 9,603 | 0 | ✅ OK |
| 2 | `account` | `ACCOUNT` | 8,144 | 0 | 8,144 | 8,144 | 0 | ✅ OK |
| 3 | `ecbs` | `ECBS` | 4,126 | 0 | 4,126 | 4,126 | 0 | ✅ OK |
| 4 | `solban` | `SOLBAN` | 522 | 0 | 522 | 522 | 0 | ✅ OK |
| 5 | `cases` | `CASES` | 6,972 | 0 | 6,972 | 6,972 | 0 | ✅ OK |
| 6 | `email_results` | `EMAIL_RESULTS` | 4,770 | 0 | 4,770 | 4,770 | 0 | ✅ OK |
| 7 | `activity_history` | `ACTIVITY_HISTORY` | 550 | 0 | 550 | 550 | 0 | ✅ OK |
| 8 | `opp_field_history` | `OPP_FIELD_HISTORY` | 140,083 | 0 | 140,083 | 140,083 | 0 | ✅ OK |
| 9 | `pagos` | `PAGOS` | 7,161 | 0 | 7,161 | 7,161 | 0 | ✅ OK |
| 10 | `stage_history` | `STAGE_HISTORY` | 53,286 | 0 | 53,286 | 53,286 | 0 | ✅ OK |

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
*Generado automáticamente el 2026-03-19 09:46 UTC*