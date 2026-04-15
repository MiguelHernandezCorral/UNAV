import sys
sys.path.insert(0, 'src')
from dotenv import load_dotenv; load_dotenv()
from oracle_connector import OracleConnector
import pandas as pd
from datetime import datetime, timedelta

conn = OracleConnector()
hoy = datetime.now()
hace4dias = hoy - timedelta(days=4)

df_pred = pd.DataFrame(conn.read_table('PMAT_PREDICTION'))
df_pred['FECHA_ACTUALIZACION'] = pd.to_datetime(df_pred['FECHA_ACTUALIZACION'], errors='coerce')
recientes = df_pred[df_pred['FECHA_ACTUALIZACION'] >= hace4dias]
print('=== PMAT_PREDICTION ===')
print(f'Total registros    : {len(df_pred):,}')
print(f'Actualizados 4d    : {len(recientes):,}')
print(f'Prob media (todos) : {pd.to_numeric(df_pred["PROBABILIDAD"], errors="coerce").mean():.1f}')
print(f'Prob media (4d)    : {pd.to_numeric(recientes["PROBABILIDAD"], errors="coerce").mean():.1f}')
print()

df_log = pd.DataFrame(conn.read_table('PMAT_SF_SYNC_LOG'))
df_log['FECHA_ENVIO'] = pd.to_datetime(df_log['FECHA_ENVIO'], errors='coerce')
log4d = df_log[df_log['FECHA_ENVIO'] >= hace4dias]
print('=== PMAT_SF_SYNC_LOG (últimos 4 días) ===')
print(f'Total envíos       : {len(log4d):,}')
print(f'OK                 : {(log4d["STATUS"]=="OK").sum():,}')
print(f'ERROR              : {(log4d["STATUS"]=="ERROR").sum():,}')
print()

log4d = log4d.copy()
log4d['DIA'] = log4d['FECHA_ENVIO'].dt.date
print('Envíos por día:')
print(log4d.groupby('DIA')['STATUS'].value_counts().unstack(fill_value=0).to_string())
print()

errores = log4d[log4d['STATUS']=='ERROR']
if len(errores):
    print(f'Tipos de error ({len(errores)} total):')
    print(errores['DETALLE'].str[:80].value_counts().head(5).to_string())
else:
    print('Sin errores en los últimos 4 días.')
