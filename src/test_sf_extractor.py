"""
test_sf_extractor.py
====================
Pruebas para sf_extractor.py.

Hay dos modos de ejecución:
  1. Sin credenciales reales → usa mocks (unittest.mock) para simular la API.
  2. Con credenciales reales  → ejecuta la llamada real si SF_CLIENT_ID está configurado.

Ejecutar:
    python src/test_sf_extractor.py
    python src/test_sf_extractor.py --real    # requiere .env con credenciales válidas
"""

import sys
import json
import unittest
from unittest.mock import patch, MagicMock

# Añadir src/ al path para importar el módulo
sys.path.insert(0, "src")
from sf_extractor import SalesforceExtractor


# ---------------------------------------------------------------------------
# Datos de ejemplo que simulan una respuesta real de la REST API de Salesforce
# ---------------------------------------------------------------------------
MOCK_AUTH_RESPONSE = {
    "access_token": "TOKEN_DE_PRUEBA_12345",
    "instance_url": "https://unav.my.salesforce.com",
    "token_type": "Bearer",
}

# Primera página: 2 registros + nextRecordsUrl para probar la paginación
MOCK_QUERY_PAGE_1 = {
    "totalSize": 3,
    "done": False,
    "nextRecordsUrl": "/services/data/v60.0/query/01g000000000001AAA-2000",
    "records": [
        {
            "attributes": {"type": "Opportunity", "url": "/services/data/v60.0/sobjects/Opportunity/001A"},
            "Id": "001AAAAAAAAAAAA",
            "AccountId": "ACC001",
            "TX_Num_Credencial__c": "CRED-001",
            "PL_Curso_academico__c": "2026/2027",
            "StageName": "Solicitud",
            "PL_Subetapa__c": "En proceso",
            "CH_Nacional__c": True,
            "CH_Internacional__c": False,
            "NU_Nota_media_admision__c": 8.5,
            "CloseDate": "2026-09-30",
            "DT_Fecha_creacion_SA__c": "2026-01-15T10:30:00.000+0000",
            "RecordTypeId": "012BBBBBBBBBBB",
            "RecordType": {
                "attributes": {"type": "RecordType"},
                "name": "Solicitud Admisión Grado",
            },
            "LK_Titulacion__c": "TIT001",
            "LK_Titulacion__r": {
                "attributes": {"type": "Titulacion__c"},
                "name": "Medicina",
            },
            "Campaign": {
                "attributes": {"type": "Campaign"},
                "name": "Campaña Grado 2026",
            },
            "CampaignId": "CAM001",
            "LK_Calendario_de_admision__r": None,
            "Paid_Amount__c": 500.0,
            "Paid_Percent__c": 25.0,
        },
        {
            "attributes": {"type": "Opportunity", "url": "/services/data/v60.0/sobjects/Opportunity/001B"},
            "Id": "001BBBBBBBBBBBBB",
            "AccountId": "ACC002",
            "TX_Num_Credencial__c": "CRED-002",
            "PL_Curso_academico__c": "2026/2027",
            "StageName": "Admisión académica",
            "PL_Subetapa__c": "Admitido",
            "CH_Nacional__c": False,
            "CH_Internacional__c": True,
            "NU_Nota_media_admision__c": 7.2,
            "CloseDate": "2026-09-30",
            "DT_Fecha_creacion_SA__c": "2026-02-20T09:00:00.000+0000",
            "RecordTypeId": "012CCCCCCCCCCC",
            "RecordType": {
                "attributes": {"type": "RecordType"},
                "name": "Solicitud Admisión Máster",
            },
            "LK_Titulacion__c": "TIT002",
            "LK_Titulacion__r": {
                "attributes": {"type": "Titulacion__c"},
                "name": "MBA Dirección de Empresas",
            },
            "Campaign": None,
            "CampaignId": None,
            "LK_Calendario_de_admision__r": None,
            "Paid_Amount__c": None,
            "Paid_Percent__c": None,
        },
    ],
}

# Segunda página: 1 registro, sin más páginas
MOCK_QUERY_PAGE_2 = {
    "totalSize": 3,
    "done": True,
    "records": [
        {
            "attributes": {"type": "Opportunity"},
            "Id": "001CCCCCCCCCCCCC",
            "AccountId": "ACC003",
            "TX_Num_Credencial__c": "CRED-003",
            "PL_Curso_academico__c": "2026/2027",
            "StageName": "Matrícula OOGG",
            "PL_Subetapa__c": "Formalizada",
            "CH_Nacional__c": True,
            "CH_Internacional__c": False,
            "NU_Nota_media_admision__c": 9.1,
            "CloseDate": "2026-09-30",
            "DT_Fecha_creacion_SA__c": "2026-03-01T08:00:00.000+0000",
            "RecordTypeId": "012DDDDDDDDDDD",
            "RecordType": {
                "attributes": {"type": "RecordType"},
                "name": "Solicitud Matricula Grado",
            },
            "LK_Titulacion__r": {"attributes": {"type": "Titulacion__c"}, "name": "Derecho"},
            "Campaign": None,
            "CampaignId": None,
            "LK_Calendario_de_admision__r": None,
            "Paid_Amount__c": 2000.0,
            "Paid_Percent__c": 100.0,
        },
    ],
}


# ---------------------------------------------------------------------------
# Helper para construir el mock de requests
# ---------------------------------------------------------------------------
def _make_mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.raise_for_status = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# Suite de tests con mock
# ---------------------------------------------------------------------------
class TestSalesforceExtractorMock(unittest.TestCase):

    def _build_sf(self) -> SalesforceExtractor:
        """Crea un SalesforceExtractor con variables de entorno simuladas."""
        with patch.dict("os.environ", {
            "SF_URL": "https://login.salesforce.com",
            "SF_SITE": "",
            "SF_CLIENT_ID": "fake_client_id",
            "SF_CLIENT_SECRET": "fake_client_secret",
            "SF_API_VERSION": "60.0",
        }):
            return SalesforceExtractor()

    # --- Autenticación ---

    def test_autenticacion_ok(self):
        """Verifica que se obtiene el token y la instance_url correctamente."""
        sf = self._build_sf()
        with patch("requests.post", return_value=_make_mock_response(MOCK_AUTH_RESPONSE)):
            sf.authenticate()
        self.assertEqual(sf._access_token, "TOKEN_DE_PRUEBA_12345")
        self.assertEqual(sf._instance_url, "https://unav.my.salesforce.com")
        print("✅ test_autenticacion_ok")

    def test_autenticacion_falla_http(self):
        """Si el servidor devuelve error HTTP, debe lanzar excepción."""
        sf = self._build_sf()
        mock_resp = _make_mock_response({}, status_code=401)
        mock_resp.raise_for_status.side_effect = Exception("401 Unauthorized")
        with patch("requests.post", return_value=mock_resp):
            with self.assertRaises(Exception):
                sf.authenticate()
        print("✅ test_autenticacion_falla_http")

    def test_config_incompleta(self):
        """Debe lanzar EnvironmentError si faltan credenciales."""
        with patch.dict("os.environ", {
            "SF_URL": "https://login.salesforce.com",
            "SF_CLIENT_ID": "",
            "SF_CLIENT_SECRET": "",
        }, clear=True):
            with self.assertRaises(EnvironmentError):
                SalesforceExtractor()
        print("✅ test_config_incompleta")

    # --- Paginación ---

    def test_paginacion_dos_paginas(self):
        """Verifica que se descargan registros de varias páginas."""
        sf = self._build_sf()
        sf._access_token = "TOKEN_DE_PRUEBA_12345"
        sf._instance_url = "https://unav.my.salesforce.com"

        respuestas = [
            _make_mock_response(MOCK_QUERY_PAGE_1),
            _make_mock_response(MOCK_QUERY_PAGE_2),
        ]
        with patch("requests.get", side_effect=respuestas):
            records = sf.query("select Id from Opportunity")

        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["Id"], "001AAAAAAAAAAAA")
        self.assertEqual(records[2]["Id"], "001CCCCCCCCCCCCC")
        print("✅ test_paginacion_dos_paginas — 3 registros recuperados en 2 páginas")

    # --- Aplanado de JSON ---

    def test_flatten_relacion_anidada(self):
        """Los campos de relación __r deben aplanarse correctamente."""
        registro = {
            "attributes": {"type": "Opportunity"},
            "Id": "001AAA",
            "RecordType": {"attributes": {"type": "RecordType"}, "name": "Grado"},
            "LK_Titulacion__r": {"attributes": {"type": "Titulacion__c"}, "name": "Medicina"},
        }
        flat = SalesforceExtractor._flatten_record(registro)

        self.assertNotIn("attributes", flat)
        self.assertIn("RecordType.name", flat)
        self.assertEqual(flat["RecordType.name"], "Grado")
        self.assertIn("LK_Titulacion__r.name", flat)
        self.assertEqual(flat["LK_Titulacion__r.name"], "Medicina")
        print("✅ test_flatten_relacion_anidada")

    def test_flatten_relacion_nula(self):
        """Si una relación es None, el campo Id debe estar presente pero sin subniveles."""
        registro = {"Id": "001AAA", "Campaign": None, "CampaignId": "CAM001"}
        flat = SalesforceExtractor._flatten_record(registro)
        self.assertEqual(flat["Campaign"], None)
        self.assertEqual(flat["CampaignId"], "CAM001")
        print("✅ test_flatten_relacion_nula")

    # --- DataFrame ---

    def test_query_to_dataframe_shape(self):
        """El DataFrame debe tener tantas filas como registros totales."""
        sf = self._build_sf()
        sf._access_token = "TOKEN_DE_PRUEBA_12345"
        sf._instance_url = "https://unav.my.salesforce.com"

        respuestas = [
            _make_mock_response(MOCK_QUERY_PAGE_1),
            _make_mock_response(MOCK_QUERY_PAGE_2),
        ]
        with patch("requests.get", side_effect=respuestas):
            raw, df = sf.query_to_dataframe("select Id from Opportunity")

        self.assertEqual(len(raw), 3)
        self.assertEqual(len(df), 3)
        print(f"✅ test_query_to_dataframe_shape — shape: {df.shape}")

    def test_columnas_aplanadas_en_df(self):
        """Las columnas de relación deben aparecer aplanadas en el DataFrame."""
        sf = self._build_sf()
        sf._access_token = "TOKEN_DE_PRUEBA_12345"
        sf._instance_url = "https://unav.my.salesforce.com"

        # Solo primera página
        pagina = {**MOCK_QUERY_PAGE_1, "done": True}
        pagina.pop("nextRecordsUrl", None)
        with patch("requests.get", return_value=_make_mock_response(pagina)):
            _, df = sf.query_to_dataframe("select Id from Opportunity")

        self.assertIn("RecordType.name", df.columns)
        self.assertIn("LK_Titulacion__r.name", df.columns)
        self.assertNotIn("attributes", df.columns)
        print("✅ test_columnas_aplanadas_en_df")
        print("   Columnas con relaciones aplanadas:", [c for c in df.columns if "." in c])

    def test_formato_fechas(self):
        """Las columnas DT_ y CloseDate deben convertirse a datetime."""
        sf = self._build_sf()
        sf._access_token = "TOKEN_DE_PRUEBA_12345"
        sf._instance_url = "https://unav.my.salesforce.com"

        pagina = {**MOCK_QUERY_PAGE_1, "done": True}
        pagina.pop("nextRecordsUrl", None)
        with patch("requests.get", return_value=_make_mock_response(pagina)):
            _, df = sf.query_to_dataframe("select Id from Opportunity")

        import pandas as pd
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["CloseDate"]),
                        "CloseDate debería ser datetime")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["DT_Fecha_creacion_SA__c"]),
                        "DT_Fecha_creacion_SA__c debería ser datetime")
        print("✅ test_formato_fechas")

    def test_query_sin_resultados(self):
        """Una consulta sin resultados debe devolver un DataFrame vacío."""
        sf = self._build_sf()
        sf._access_token = "TOKEN_DE_PRUEBA_12345"
        sf._instance_url = "https://unav.my.salesforce.com"

        respuesta_vacia = {"totalSize": 0, "done": True, "records": []}
        with patch("requests.get", return_value=_make_mock_response(respuesta_vacia)):
            raw, df = sf.query_to_dataframe("select Id from Opportunity where Id = 'INEXISTENTE'")

        self.assertEqual(len(raw), 0)
        self.assertTrue(df.empty)
        print("✅ test_query_sin_resultados")


# ---------------------------------------------------------------------------
# Prueba contra la API real (solo si --real está en los args)
# ---------------------------------------------------------------------------
def run_real_test():
    import os
    from dotenv import load_dotenv
    load_dotenv()

    print("\n" + "=" * 60)
    print("MODO REAL — conectando a Salesforce con credenciales del .env")
    print("=" * 60)

    try:
        sf = SalesforceExtractor()
        sf.authenticate()
        print(f"✅ Token obtenido. Instance URL: {sf._instance_url}")

        # Consulta reducida para validar la conexión rápidamente
        soql_test = """
            select Id, StageName, PL_Curso_academico__c, RecordType.name
            from Opportunity
            where PL_Curso_academico__c = '2026/2027'
            LIMIT 5
        """
        raw, df = sf.query_to_dataframe(soql_test)

        print(f"\nRegistros recibidos: {len(raw)}")
        print("\n--- DataFrame ---")
        print(df.to_string(index=False))

        # Guardar resultado para inspección
        df.to_csv("test_real_output.csv", sep=";", index=False, encoding="utf-8-sig")
        with open("test_real_output.json", "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2, default=str)
        print("\n✅ Resultados guardados en test_real_output.csv y test_real_output.json")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if "--real" in sys.argv:
        sys.argv.remove("--real")
        run_real_test()
    else:
        print("=" * 60)
        print("Ejecutando tests con MOCK (sin credenciales reales)")
        print("Usa --real para probar contra la API de Salesforce")
        print("=" * 60 + "\n")
        unittest.main(verbosity=0)
