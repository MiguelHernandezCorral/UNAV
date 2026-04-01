"""
sf_extractor.py
===============
Extrae datos de Salesforce vía REST API (OAuth2 client_credentials),
ejecuta una consulta SOQL, aplana los JSON anidados y devuelve un DataFrame.

Uso:
    from sf_extractor import SalesforceExtractor
    sf = SalesforceExtractor()
    df_raw, df = sf.query_to_dataframe(SOQL_QUERY)
"""

from __future__ import annotations

import os
import re
import logging
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Consulta SOQL principal (copiada del archivo 1- Opportunity.sql)
# ---------------------------------------------------------------------------
SOQL_OPPORTUNITY = """
select Id,
    AccountId,
    TX_Num_Credencial__c,
    PL_Curso_academico__c,
    PL_Mes_Anio_inicio__c,
    PL_Tipo_Acceso__c,
    RecordTypeId,
    RecordType.name,
    PL_Estado__c,
    StageName,
    PL_Subetapa__c,
    Estado_Matricula__c,
    CH_Simultaneidad__c,
    CH_Completa__c,
    Cambia_de_carrera__c,
    PL_Gestionado_por__c,
    CH_Nacional__c,
    CH_Internacional__c,
    PL_Origen_de_solicitud__c,
    PL_Formato__c,
    CH_Origen_admision__c,
    CloseDate,
    LeadSource,
    PL_Origen_formulario__c,
    PL_Plazo_admision__c,
    LK_Calendario_de_admision__c,
    LK_Calendario_de_admision__r.name,
    LK_Calendario_de_admision_final__c,
    LK_Calendario_de_admision_final__r.name,
    ConvocatoriaNombre__c,
    NU_Nota_media_admision__c,
    CH_Pruebas_calificadas__c,
    NU_Resultado_admision_puntos__c,
    CH_StandBy__c,
    FOR_Estado_tipo_documento__c,
    DT_Fecha_propuesta_resolucion_facultad__c,
    DT_Nueva_fecha_resolucion_candidato__c,
    NU_Preferencia__c,
    LK_Titulacion__c,
    LK_Titulacion__r.name,
    LK_Titulacion_definitiva__c,
    LK_Titulacion_definitiva__r.name,
    LK_Titulacion_por_curso_academico__c,
    LK_Titulacion_por_curso_academico__r.name,
    LK_Titulacion_por_curso_academico_final__c,
    LK_Titulacion_por_curso_academico_final__r.name,
    PL_Resolucion_definitiva__c,
    PL_Resolucion_inicial__c,
    CH_Titulacion_reorientacion__c,
    RES_Titulaciones_reorientacion__c,
    CH_Carta_enviada__c,
    TX_Observaciones_facultad__c,
    TX_Observaciones_delegado__c,
    ObservacionesAdmision__c,
    TX_Observaciones_admision__c,
    Description__c,
    ObservacionesDeCondicionado__c,
    ObservacionesDeDenegado__c,
    PL_Motivo_cancelacion__c,
    TX_Motivo_cancelacion__c,
    Observacionesmaster__c,
    Observacionesresolucion__c,
    PL_Curso_actual__c,
    LK_CentroEnsenanza__c,
    LK_CentroEnsenanza__r.name,
    TX_Pais_CentroEnsenanza__c,
    TX_Provincia_CentroEnsenanza__c,
    TX_Localidad_CentroEnsenanza__c,
    TX_OtroCentroEnsenanza__c,
    PL_Modalidad_de_acceso__c,
    PL_Selectividad_UPNA_TrasladoEXP__c,
    CH_Matricula_de_curso_anterior__c,
    CU_Importe_total__c,
    ImportePrimerPago__c,
    FechaLimitePrimerPago__c,
    MinimumPaymentPayed__c,
    MinimumPaymentDate__c,
    Paid_Amount__c,
    Paid_Percent__c,
    CH_Pago_superior__c,
    Importe_minimo_personalizado__c,
    LK_Descuento_Pronto_Pago__c,
    NU_Importe_descuento_pronto_pago__c,
    NU_Porcentaje_descuento_pronto_pago__c,
    DT_Fecha_creacion_SA__c,
    DT_Fecha_envio_SA__c,
    DT_Fecha_Publicacion_resolucion__c,
    DT_Fech_fin_plazo__c,
    TX_Beca_solicitada__c,
    CH_Matricula_sujeta_beca__c,
    Motivomaster__c,
    TX_Motivo_solicitud__c,
    PL_Valoracion_estudio_postgrado__c,
    Fech_Definitiva_Valoraci_n__c,
    TX_Motivo_valoracion__c,
    CampaignId,
    Campaign.name,
    Signature__c,
    Informe_Matricula_Firmado_Attachment__c,
    ProbabilidadMatriculaDelegado__c,
    DT_Dia_hora_cita__c,
    CH_Informacion_enviada__c,
    CH_Conditioned__c,
    CH_Solicita_Alojamiento__c,
    noContabilizable__c,
    PL_Modalidad_examen_acceso_escogido__c,
    Kitdevisado__c,
    FechenvioKit__c,
    DT_Fecha_inicio_Matricula_admision__c,
    DT_Fecha_fin_Matricula_admision__c,
    DT_Fecha_inicio_Matricula_OOGG__c,
    DT_Fecha_fin_matricula_OOGG__c,
    NU_PasoMax__c,
    NU_Paso_solicitud_admision__c,
    CH_Acepto_condiciones_matricula__c,
    DT_Fecha_Acepta_cond_matr__c,
    PL_Domicilio_durante_curso__c,
    LK_Colegio_M_Residencia_durante_curso__c,
    LK_Colegio_M_Residencia_durante_curso__r.name,
    OwnerId
from Opportunity
where PL_Curso_academico__c IN ('2025/2026', '2026/2027')
and RecordType.name in ('Solicitud Admisión Grado', 'Solicitud Admisión Máster', 'Solicitud Matricula Grado', 'Solicitud Matricula Máster')
"""


class SalesforceExtractor:
    """
    Cliente para la REST API de Salesforce.

    Variables de entorno requeridas (o en fichero .env):
        SF_URL          – URL base de Salesforce, p.ej. https://login.salesforce.com
        SF_SITE         – Sufijo del site si lo hay (dejar vacío si no aplica)
        SF_CLIENT_ID    – Consumer Key de la Connected App
        SF_CLIENT_SECRET– Consumer Secret de la Connected App
        SF_API_VERSION  – Versión de la API, p.ej. 60.0
    """

    def __init__(self):
        self.url = os.getenv("SF_URL", "").rstrip("/")
        self.site = os.getenv("SF_SITE", "")
        self.client_id = os.getenv("SF_CLIENT_ID", "")
        self.client_secret = os.getenv("SF_CLIENT_SECRET", "")
        self.api_version = os.getenv("SF_API_VERSION", "60.0")

        self._access_token: str | None = None
        self._instance_url: str | None = None

        self._validate_config()

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        missing = [
            var for var, val in {
                "SF_URL": self.url,
                "SF_CLIENT_ID": self.client_id,
                "SF_CLIENT_SECRET": self.client_secret,
            }.items() if not val
        ]
        if missing:
            raise EnvironmentError(
                f"Faltan variables de entorno de Salesforce: {', '.join(missing)}"
            )

    # ------------------------------------------------------------------
    # Autenticación OAuth2
    # ------------------------------------------------------------------

    def authenticate(self) -> None:
        """Obtiene el Bearer token usando grant_type=client_credentials."""
        token_url = f"{self.url}{self.site}/services/oauth2/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        logger.info("Autenticando en Salesforce: %s", token_url)
        response = requests.post(token_url, data=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        self._access_token = data["access_token"]
        self._instance_url = data.get("instance_url", self.url)
        logger.info("Autenticación correcta. Instance URL: %s", self._instance_url)

    @property
    def _headers(self) -> dict:
        if not self._access_token:
            self.authenticate()
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Consulta SOQL con paginación automática
    # ------------------------------------------------------------------

    def query(self, soql: str) -> list[dict]:
        """
        Ejecuta una consulta SOQL y devuelve todos los registros (gestiona
        la paginación automática de Salesforce, que limita a 2000 por página).

        Returns:
            Lista de dicts con los registros en crudo.
        """
        # Eliminar saltos de línea para que la URL sea válida
        soql_clean = re.sub(r"\s+", " ", soql).strip()
        endpoint = (
            f"{self._instance_url}/services/data/v{self.api_version}"
            f"/query/?q={requests.utils.quote(soql_clean)}"
        )

        records: list[dict] = []
        next_url: str | None = endpoint

        while next_url:
            logger.info("Consultando: %s", next_url[:120] + "...")
            resp = requests.get(next_url, headers=self._headers, timeout=60)
            resp.raise_for_status()
            body = resp.json()

            batch = body.get("records", [])
            records.extend(batch)
            logger.info(
                "Registros recibidos: %d / total: %s",
                len(batch),
                body.get("totalSize", "?"),
            )

            # Paginación: Salesforce incluye nextRecordsUrl si hay más páginas
            raw_next = body.get("nextRecordsUrl")
            if raw_next:
                next_url = f"{self._instance_url}{raw_next}"
            else:
                next_url = None

        logger.info("Total registros descargados: %d", len(records))
        return records

    # ------------------------------------------------------------------
    # Aplanado de JSON anidado
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_record(record: dict, parent_key: str = "", sep: str = ".") -> dict:
        """
        Convierte un dict anidado (p.ej. registros con relaciones __r o Campaign)
        en un dict plano usando la notación padre.hijo como nombre de columna.

        Salesforce añade un campo 'attributes' en cada objeto; lo eliminamos.
        """
        flat: dict = {}
        for key, value in record.items():
            if key == "attributes":
                continue
            full_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                # Nodo de relación anidada → aplanar recursivamente
                nested = SalesforceExtractor._flatten_record(value, full_key, sep)
                flat.update(nested)
            else:
                flat[full_key] = value
        return flat

    # ------------------------------------------------------------------
    # Conversión a DataFrame con formateo básico
    # ------------------------------------------------------------------

    @staticmethod
    def _format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica un formateo básico al DataFrame:
        - Convierte columnas de fecha a datetime.
        - Normaliza checkboxes booleanos.
        - Elimina el prefijo 'attributes' residual si existiera.
        """
        # Detectar y convertir columnas de fecha (patrón SF: DT_ / Date / _date)
        date_patterns = re.compile(
            r"(^DT_|Date$|_date$|CloseDate|MinimumPaymentDate|"
            r"FechaLimitePrimerPago|Fech_Definitiva|FechenvioKit)",
            re.IGNORECASE,
        )
        for col in df.columns:
            if date_patterns.search(col):
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        # Normalizar booleanos: Salesforce puede devolver True/False o 'true'/'false'
        for col in df.select_dtypes(include="object").columns:
            lowered = df[col].dropna().str.lower()
            if set(lowered.unique()).issubset({"true", "false"}):
                df[col] = df[col].map({"true": True, "True": True, "false": False, "False": False})

        return df

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def query_to_dataframe(
        self, soql: str, flatten: bool = True
    ) -> tuple[list[dict], pd.DataFrame]:
        """
        Ejecuta la consulta SOQL y devuelve:
            - records_raw : lista de dicts con el JSON original de SF
            - df          : DataFrame aplanado y formateado

        Parámetros:
            soql    – Consulta SOQL
            flatten – Si True (default), aplana los campos de relación anidados
        """
        records_raw = self.query(soql)

        if not records_raw:
            logger.warning("La consulta no devolvió registros.")
            return records_raw, pd.DataFrame()

        if flatten:
            flat_records = [self._flatten_record(r) for r in records_raw]
        else:
            flat_records = records_raw

        df = pd.DataFrame(flat_records)
        df = self._format_dataframe(df)

        logger.info("DataFrame creado: %d filas x %d columnas", *df.shape)
        return records_raw, df


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    sf = SalesforceExtractor()

    records_raw, df = sf.query_to_dataframe(SOQL_OPPORTUNITY)

    # Guardar JSON crudo
    output_json = "opportunity_raw.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records_raw, f, ensure_ascii=False, indent=2, default=str)
    logger.info("JSON guardado en: %s", output_json)

    # Guardar CSV formateado
    output_csv = "opportunity.csv"
    df.to_csv(output_csv, sep=";", index=False, encoding="utf-8-sig")
    logger.info("CSV guardado en: %s", output_csv)

    print("\n--- Primeras filas del DataFrame ---")
    print(df.head())
    print("\n--- Columnas ---")
    print(df.columns.tolist())
    print(f"\nShape: {df.shape}")
