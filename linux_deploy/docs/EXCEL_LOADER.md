# Carga Histórica – Excel → Oracle

## ¿Qué hace este script?

Lee el fichero histórico de Salesforce `DataSET_SF - V2.xlsx` y carga
5 hojas en Oracle como tablas históricas (`*_HIST`). Estas tablas se
combinan con los datos actuales de Salesforce (curso 2026/2027) en la
fase de limpieza (`cleaner.py --include-historical`) para entrenar el
modelo con datos históricos reales y predecir el curso vigente.

## Diferencias entre tablas Oracle API y tablas HIST

| Aspecto | Oracle API (SF 2026/27) | Tablas HIST (Excel histórico) |
|---------|-------------------------|-------------------------------|
| Nombres columnas | Formato API (`PL_CURSO_ACADEMICO__C`) | Formato notebook (`PL_CURSO_ACADEMICO`) |
| Join Account | `ACCOUNTID → ID` (15 chars) | `ACCOUNTID → ID18` (18 chars) |
| YEARPERSONBIRTHDATE | Se deriva de `PERSONBIRTHDATE` | Ya calculado en Excel |
| CENTROENSENANZA | De OPPORTUNITY (relación anidada) | De ACCOUNT_HIST (columna directa) |
| CH_ANTIGUO_ALUMNO__PC | Renombrado de `CH_ANTIGUOALUMNO__PC` | Nombre correcto directo |

## Ejecución

- **Fichero fuente:** `C:\Users\jvelazquezc\Downloads\DataSET_SF - V2.xlsx`
- **Fecha/hora inicio:** 2026-03-19 11:50:39 UTC
- **Fecha/hora fin:**    2026-03-19 12:03:36 UTC
- **Duración total:**    777.6 s
- **Modo:**              RECREAR tablas (--recreate)

## Métricas por tabla

| Tabla Oracle | Hoja Excel | Filas Excel | DB antes | DB después | Insertados |
|---|---|---|---|---|---|
| `OPPORTUNITY_HIST` | Oportunidad_OK | 70,297 | 0 | 70,297 | 70,297 |
| `ACCOUNT_HIST` | Cuenta | 55,275 | 0 | 55,275 | 55,275 |
| `ECBS_HIST` | ECB | 6,282 | 0 | 6,282 | 6,282 |
| `STAGE_HISTORY_HIST` | Historial_etapas_Oportunidad_OK | 536,575 | 0 | 536,575 | 536,575 |
| `ACTIVITY_HIST` | Historial_actividad_promocion | 62,170 | 0 | 62,170 | 62,170 |

## Tablas creadas

| Tabla | Descripción | PK |
|---|---|---|
| `OPPORTUNITY_HIST` | Oportunidades históricas (múltiples cursos) | `ID` (upsert) |
| `ACCOUNT_HIST` | Cuentas/personas históricas | `ID18` (upsert) |
| `ECBS_HIST` | Estudios coste y becas históricos | `Id` (upsert) |
| `STAGE_HISTORY_HIST` | Historial de etapas histórico (536 K filas) | — (insert) |
| `ACTIVITY_HIST` | Historial de actividades de promoción | — (insert) |

## Uso en cleaner.py

Una vez cargadas las tablas HIST, ejecutar:

```bash
python src/cleaner.py --recreate --include-historical
```

Esto combina:
- Datos actuales de SF (2026/27): ~9 600 oportunidades × etapas
- Datos históricos del Excel: ~70 000 oportunidades × etapas

- **Log completo:** `logs/excel_loader.log`

---
*Generado automáticamente el 2026-03-19 12:03 UTC*