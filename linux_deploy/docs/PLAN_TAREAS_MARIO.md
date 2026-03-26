# Plan de tareas — Mario Almendros
**Fecha:** 26-mar-2026
**Objetivo:** familiarizarse con el código del pipeline e ir haciendo análisis y pequeñas implementaciones de forma autónoma.

Las tareas están organizadas de menos a más difícil. No hace falta hacerlas todas seguidas — ve a tu ritmo y pregunta lo que necesites.

---

## BLOQUE A — Terminar y limpiar el notebook actual

**A1 — Quitar las credenciales del notebook** ⚠️ *Prioritario*
En la celda 3 tienes `DB_PASS = "w6IT%_)M>&"` escrito directamente. Hay que eliminarlo porque queda registrado en el historial de git aunque lo borres después. La solución es eliminar esas primeras celdas de conexión directa y quedarte solo con el `OracleConnector()` que ya usas más abajo, que lee las credenciales del fichero `.env` automáticamente.

**A2 — Corregir el bug del triple-quote en celda 16**
Hay un bloque de código entre `"""..."""` que lo convierte en un string — no se ejecuta. Dentro de ese bloque están `etapa_counts` y `etapa_pct`, que luego se usan fuera. Eso daría `NameError` al ejecutar el notebook desde cero. Saca esos cálculos fuera del string o elimina las comillas.

**A3 — Distinguir Grado/Máster sin JOIN**
Intentaste hacer un JOIN con la tabla ACCOUNT pero la clave no encajaba. La solución más sencilla es que la columna `MODELO` de `PMAT_PREDICTION` ya tiene `grado_v1` o `master_v1`. Con eso puedes filtrar directamente:
```python
df_grado  = df[df['MODELO'].str.startswith('grado')]
df_master = df[df['MODELO'].str.startswith('master')]
```
Muestra la distribución de ETAPA separada para cada uno.

**A4 — Explicar los nulos altos en variables clave**
Variables como `CU_IMPORTE_TOTAL` o `CH_MATRICULA_SUJETA_BECA` tienen >98% de nulos. Busca en `src/cleaner.py` la constante `COLS_NUNCA_ELIMINAR` y lee el comentario que la acompaña. Añade en el notebook una celda markdown explicando por qué estos nulos son esperables y no un problema.

---

## BLOQUE B — Análisis de PROBABILIDAD

**B1 — Probabilidad media por ETAPA**
Calcula la media de `PROBABILIDAD` agrupando por `ETAPA`. ¿Hay etapas donde el modelo es sistemáticamente más alto o más bajo? ¿Tiene sentido de negocio?

**B2 — Boxplot de PROBABILIDAD por ETAPA**
Dibuja un boxplot para las 8-10 etapas más frecuentes. Verás cómo se distribuye la incertidumbre del modelo en cada punto del funnel.
```python
import matplotlib.pyplot as plt
top_etapas = df['ETAPA'].value_counts().head(8).index
df[df['ETAPA'].isin(top_etapas)].boxplot(column='PROBABILIDAD', by='ETAPA', figsize=(12,5), rot=45)
plt.tight_layout(); plt.show()
```

**B3 — Usar PMAT_PRED_ACTUAL en vez de PMAT_PREDICTION**
La vista `PMAT_PRED_ACTUAL` tiene una sola fila por candidato (la etapa más reciente). Repite el análisis anterior con ella y compara. ¿Cambia mucho la distribución?
```python
df_actual = pd.DataFrame(conn.read_table('PMAT_PRED_ACTUAL'))
```

**B4 — Segmentación por umbral**
Calcula cuántos candidatos superan los umbrales 50%, 70% y 90% de probabilidad. ¿Ese número de "casi seguros" parece razonable respecto al histórico de matrículas?

---

## BLOQUE C — Entender el código del pipeline

**C1 — Leer preprocessor.py**
Abre `src/preprocessor.py` y localiza la función `calcular_orden_automatico()`. Escribe en una celda markdown del notebook qué hace exactamente: ¿qué columna calcula?, ¿para qué la usa el modelo?

**C2 — Entender CONFIANZA**
En `src/predictor.py`, busca cómo se calcula `CONFIANZA` (pista: está en `construir_resultado_v2`). ¿Qué significa un valor de 20 vs uno de 80? Escribe una explicación en lenguaje de negocio.

**C3 — Ejecutar el pipeline en dry-run**
Con el entorno virtual activado, ejecuta:
```bash
python src/pipeline.py --phases fase3 --dry-run
```
Observa los logs y escribe en el notebook qué hace el `--dry-run` en fase3. Pista: no se comporta igual que en fase1 o fase2 — mira el código de `pipeline.py` para entender por qué.

---

## BLOQUE D — Verificar el write-back a Salesforce

**D1 — Tasa de éxito del write-back**
Consulta la tabla `PMAT_SF_SYNC_LOG` y calcula cuántos envíos salieron OK y cuántos con ERROR en la última ejecución. ¿Qué porcentaje de error hay?
```python
df_log = pd.DataFrame(conn.read_table('PMAT_SF_SYNC_LOG'))
df_log['FECHA_ENVIO'] = pd.to_datetime(df_log['FECHA_ENVIO'])
ultimo_dia = df_log.sort_values('FECHA_ENVIO').tail(10000)
print(ultimo_dia['STATUS'].value_counts())
```

**D2 — Analizar los errores**
Para los registros con `STATUS='ERROR'`, inspecciona el campo `DETALLE`. ¿Qué tienen en común los que fallan? ¿Son oportunidades cerradas, campos obligatorios, etc.?

**D3 — Envíos por ETAPA**
Cruza `PMAT_SF_SYNC_LOG` con `PMAT_PRED_ACTUAL` (por `OPP_ID`) y muestra cuántos candidatos de cada ETAPA fueron enviados con éxito. ¿Alguna etapa falla más que otras?

---

## BLOQUE E — Primera implementación propia

**E1 — Localizar dónde añadir CONFIANZA en el write-back**
Abre `src/sf_writer.py` y localiza la función `_enviar_lote()`. El campo `CONFIANZA` no se envía a Salesforce todavía (pendiente de que el cliente confirme el nombre del campo). Escribe en un comentario del notebook exactamente qué líneas habría que cambiar para añadirlo cuando llegue esa confirmación.

**E2 — Completar sync_deploy.bat**
El fichero `sync_deploy.bat` en la raíz del proyecto copia los ficheros de `src/` a `linux_deploy/src/`. Ábrelo y comprueba si `sf_writer.py` está incluido. Si no, añádelo siguiendo el mismo patrón que las demás líneas.

**E3 — *(Reto)* Escribir un test para sf_writer.py**
Crea un nuevo fichero `src/test_sf_writer.py` con al menos dos tests:
1. Que `_enviar_lote()` construye el payload con `allOrNone: False` y el campo `NU_Probabilidad_de_matricula__c` con el valor correcto.
2. Que `sf_writer.run(dry_run=True)` no llama a `requests.patch`.

Usa `unittest.mock.patch` para no hacer llamadas reales. Mira `src/test_phase5_predicciones_v2.py` como referencia de cómo están hechos los mocks.

---

*Cualquier duda, pregunta sin problema — es normal que haya partes del código que no estén claras al principio.*
