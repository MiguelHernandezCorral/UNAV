# Fase 2 – Limpieza y construcción del dataset de modelado

## ¿Qué hace esta fase?

Lee las 10 tablas de Oracle cargadas en Fase 1, aplica la lógica de limpieza
del notebook `01_Limpieza copy.ipynb` y genera la tabla `DATASET_LIMPIO`
lista para modelado con PyCaret.

### Pipeline de limpieza (15 pasos)

| Paso | Descripción |
|------|-------------|
| 1  | Carga de tablas desde Oracle (OPPORTUNITY, ACCOUNT, ECBS, STAGE_HISTORY, ACTIVITY_HISTORY) |
| 2  | Renombrado de columnas Oracle → convención notebook |
| 3  | Validación de campos: notebook vs Oracle |
| 4  | Eliminación de columnas con >90% de NA |
| 5  | Creación del target (Matrícula OOGG formalizada) y flag desmatriculado |
| 6  | JOIN Opportunity × Account (LEFT, clave ACCOUNTID = ID) |
| 7  | Variables derivadas: YEARPERSONBIRTHDATE, TITULACION, CENTROENSENANZA, RECORDTYPENAME, PL_TIPOSOLICITUD |
| 8  | Normalización del plazo de admisión: Diciembre / Marzo / Rolling / Otros |
| 9  | JOIN con ECBS + cálculo de PORCENTAJE_PAGADO_FINAL |
| 10 | Cálculo de tiempos en cada etapa del funnel (tiempo_etapa_dias, tiempo_entre_etapas_dias) |
| 11 | Control de leakage por hitos temporales (pruebas calificadas, pago mínimo) |
| 12 | Integración de actividades acumuladas (num_asistencias_acum, num_solicitudes_acum) |
| 13 | Detección y corrección de leakage de pago en etapas tempranas |
| 14 | Eliminación de registros sin target válido |
| 15 | Selección de 56 columnas finales y guardado en Oracle |

## Ejecución

- **Fecha/hora inicio:** 2026-03-19 10:20:27 UTC
- **Fecha/hora fin:**    2026-03-19 10:20:40 UTC
- **Duración total:**    12.1 s
- **Modo:**              RECREAR tabla (--recreate)

## Métricas del dataset

| Métrica | Valor |
|---------|-------|
| Filas tras join (antes limpieza) | 53,343 |
| Filas con target nulo eliminadas | 0 |
| Filas finales en DATASET_LIMPIO  | 53,343 |
| Columnas seleccionadas           | 52 |
| Target = 1 (matriculados)        | 0 |
| Target = 0 (no matriculados)     | 53,343 |
| Tasa conversión                  | 0.0 % |

## Comparativa: campos notebook vs Oracle

Esta tabla muestra si cada campo esperado por el notebook existe en Oracle,
si necesita renombrado, si es derivado o si está ausente.

| Campo notebook | Tabla Oracle | Columna Oracle | Estado | Nota |
|---|---|---|---|---|
| `ACCOUNTID` | OPPORTUNITY | `ACCOUNTID` | ✅ disponible | directo |
| `ID` | OPPORTUNITY | `ID` | ✅ disponible | directo |
| `ID18__PC` | ACCOUNT | `ID18__PC` | ✅ disponible | directo |
| `target` | COMPUTED | `—` | 🔧 derivado | derivado: Matrícula OOGG + Formalizada |
| `desmatriculado` | COMPUTED | `—` | 🔧 derivado | derivado: PL_Subetapa__c=Desmatriculado |
| `PL_CURSO_ACADEMICO` | OPPORTUNITY | `PL_CURSO_ACADEMICO__C` | ✅ disponible | renombrado |
| `CH_NACIONAL` | OPPORTUNITY | `CH_NACIONAL__C` | ✅ disponible | renombrado |
| `NU_NOTA_MEDIA_ADMISION` | OPPORTUNITY | `NU_NOTA_MEDIA_ADMISION__C` | ✅ disponible | renombrado |
| `NU_NOTA_MEDIA_1_BACH__PC` | ACCOUNT | `NU_NOTA_MEDIA_1_BACH__PC` | ✅ disponible | directo |
| `CH_PRUEBAS_CALIFICADAS` | OPPORTUNITY | `CH_PRUEBAS_CALIFICADAS__C` | ✅ disponible | renombrado |
| `NU_RESULTADO_ADMISION_PUNTOS` | OPPORTUNITY | `NU_RESULTADO_ADMISION_PUNTOS__C` | ✅ disponible | renombrado |
| `PL_RESOLUCION_DEFINITIVA` | OPPORTUNITY | `PL_RESOLUCION_DEFINITIVA__C` | ✅ disponible | renombrado |
| `TITULACION` | OPPORTUNITY | `LK_TITULACION_DEFINITIVA__R.NAME` | ✅ disponible | renombrado (relación) |
| `CENTROENSENANZA` | OPPORTUNITY | `LK_CENTROENSENANZA__R.NAME` | ✅ disponible | renombrado (relación) |
| `MINIMUMPAYMENTPAYED` | OPPORTUNITY | `MINIMUMPAYMENTPAYED__C` | ✅ disponible | renombrado |
| `PAID_AMOUNT` | OPPORTUNITY | `PAID_AMOUNT__C` | ✅ disponible | renombrado |
| `PAID_PERCENT` | OPPORTUNITY | `PAID_PERCENT__C` | ✅ disponible | renombrado |
| `CH_PAGO_SUPERIOR` | OPPORTUNITY | `CH_PAGO_SUPERIOR__C` | ✅ disponible | renombrado |
| `CH_MATRICULA_SUJETA_BECA` | OPPORTUNITY | `CH_MATRICULA_SUJETA_BECA__C` | ✅ disponible | renombrado |
| `CH_AYUDA_FINANCIACION` | ACCOUNT | `CH_AYUDA_FINANCIACION__C` | ✅ disponible | renombrado |
| `CU_IMPORTE_TOTAL` | OPPORTUNITY | `CU_IMPORTE_TOTAL__C` | ✅ disponible | renombrado |
| `CH_VISITACAMPUS__PC` | ACCOUNT | `CH_VISITACAMPUS__PC` | ✅ disponible | directo |
| `CH_ENTREVISTA_PERSONAL__PC` | ACCOUNT | `CH_ENTREVISTA_PERSONAL__PC` | ✅ disponible | directo |
| `ACC_DTT_FECHAULTIMAACTIVIDAD` | ACCOUNT | `ACC_DTT_FECHAULTIMAACTIVIDAD__C` | ✅ disponible | renombrado |
| `NU_PREFERENCIA` | OPPORTUNITY | `NU_PREFERENCIA__C` | ✅ disponible | renombrado |
| `PL_Etapa__c` | STAGE_HISTORY | `PL_ETAPA__C` | ✅ disponible | renombrado |
| `PL_Subetapa__c` | STAGE_HISTORY | `PL_SUBETAPA__C` | ✅ disponible | renombrado |
| `CH_HIJO_EMPLEADO__PC` | ACCOUNT | `CH_HIJO_EMPLEADO__PC` | ✅ disponible | directo |
| `CH_HIJO_PROFESOR_ASOCIADO__PC` | ACCOUNT | `CH_HIJO_PROFESOR_ASOCIADO__PC` | ✅ disponible | directo |
| `CH_HERMANOS_ESTUDIANDO_UNAV__P` | ACCOUNT | `CH_HERMANOS_ESTUDIANDO_UNAV__PC` | ✅ disponible | truncado en notebook |
| `CH_HIJO_MEDICO__PC` | ACCOUNT | `CH_HIJO_MEDICO__PC` | ✅ disponible | directo |
| `YEARPERSONBIRTHDATE` | ACCOUNT | `PERSONBIRTHDATE` | ✅ disponible | derivado: año de PERSONBIRTHDATE |
| `NAMEX` | — | `—` | ❌ no existe en Oracle | ⚠️ NO existe en Oracle (no se seleccionó) |
| `CH_FAMILIA_NUMEROSA__PC` | ACCOUNT | `CH_FAMILIA_NUMEROSA__PC` | ✅ disponible | directo |
| `PL_SITUACION_SOCIO_ECONOMICA` | ACCOUNT | `PL_SITUACION_SOCIO_ECONOMICA__PC` | ✅ disponible | renombrado |
| `LEADSOURCE` | OPPORTUNITY | `LEADSOURCE` | ✅ disponible | directo |
| `PL_ORIGEN_DE_SOLICITUD` | OPPORTUNITY | `PL_ORIGEN_DE_SOLICITUD__C` | ✅ disponible | renombrado |
| `PL_PLAZO_ADMISION` | OPPORTUNITY | `PL_PLAZO_ADMISION__C` | ✅ disponible | renombrado |
| `RECORDTYPENAME` | OPPORTUNITY | `RECORDTYPE.NAME` | ✅ disponible | renombrado (relación anidada) |
| `PLAZO_ADMISION_LIMPIO` | COMPUTED | `—` | 🔧 derivado | derivado: normalización PL_PLAZO_ADMISION |
| `FO_rentaFam_ges__c` | ECBS | `FO_RENTAFAM_GES__C` | ✅ disponible | renombrado |
| `CU_precioOrdinario_def__c` | ECBS | `CU_PRECIOORDINARIO_DEF__C` | ✅ disponible | renombrado |
| `CU_precioAplicado_def__c` | ECBS | `CU_PRECIOAPLICADO_DEF__C` | ✅ disponible | renombrado |
| `PORCENTAJE_PAGADO_FINAL` | COMPUTED | `—` | 🔧 derivado | derivado: precio_aplicado/precio_ordinario*100 |
| `tiempo_etapa_dias` | COMPUTED | `—` | 🔧 derivado | derivado: Fecha_fin - CreatedDate |
| `tiempo_entre_etapas_dias` | COMPUTED | `—` | 🔧 derivado | derivado: lag entre etapas consecutivas |
| `num_asistencias_acum` | COMPUTED | `—` | 🔧 derivado | derivado: actividades asistidas acumuladas |
| `num_solicitudes_acum` | COMPUTED | `—` | 🔧 derivado | derivado: solicitudes acumuladas |
| `CH_ALUMNO__PC` | ACCOUNT | `CH_ALUMNO__PC` | ✅ disponible | directo |
| `CH_ESTUDIANTE__PC` | ACCOUNT | `CH_ESTUDIANTE__PC` | ✅ disponible | directo |
| `CH_ANTIGUO_ALUMNO__PC` | ACCOUNT | `CH_ANTIGUOALUMNO__PC` | ✅ disponible | ⚠️ nombre distinto: Oracle = CH_ANTIGUOALUMNO__PC |
| `CH_ALUMNI__PC` | ACCOUNT | `CH_ALUMNI__PC` | ✅ disponible | directo |
| `CH_ANTIGUOALUMNO_INTERCAMBIO` | ACCOUNT | `CH_ANTIGUOALUMNO_INTERCAMBIO__PC` | ✅ disponible | renombrado |
| `CH_HIJO_ANTIGUO_ALUMNO__PC` | ACCOUNT | `CH_HIJO_ANTIGUO_ALUMNO__PC` | ✅ disponible | directo |
| `CreatedDate` | STAGE_HISTORY | `CREATEDDATE` | ✅ disponible | renombrado |
| `NU_MEDIA_EXPEDIENTE_UNIVERSITA` | ACCOUNT | `NU_MEDIA_EXPEDIENTE_UNIVERSITARIO__C` | ✅ disponible | ⚠️ truncado en notebook (30 chars) |

## Diferencias y campos no disponibles

- **`NAMEX`** — ❌ no existe en Oracle: ⚠️ NO existe en Oracle (no se seleccionó)

## Notas técnicas

- **Leakage académico**: las columnas de notas y resolución se ponen a NaN
  si la fila del historial es anterior a la fecha en que se calificaron las pruebas.
- **Leakage económico**: las columnas de pago se ponen a NaN si la fila es
  anterior al primer pago mínimo.
- **NAMEX**: el campo nombre no fue seleccionado en la query de Salesforce.
  Se puede añadir en fases futuras si se necesita.
- **CH_ANTIGUO_ALUMNO__PC**: en Oracle la columna se llama `CH_ANTIGUOALUMNO__PC`
  (sin guión bajo entre ANTIGUO y ALUMNO). Se renombra al cargar.
- **Truncados**: CH_HERMANOS_ESTUDIANDO_UNAV__P y NU_MEDIA_EXPEDIENTE_UNIVERSITA
  aparecen truncados en el notebook (nombres de hasta 30 chars del Excel).
  En Oracle tienen su nombre completo.
- **Log completo**: `logs/cleaner.log`

---
*Generado automáticamente el 2026-03-19 10:20 UTC*