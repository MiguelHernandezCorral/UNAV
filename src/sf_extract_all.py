"""
sf_extract_all.py
=================
FASE 1 – Ingesta Salesforce → Oracle
======================================
Extrae todas las entidades necesarias desde Salesforce y las carga
directamente en Oracle via MERGE INTO (upsert).
No se generan CSVs intermedios: los registros JSON de Salesforce se
aplanan y se envían directamente a Oracle.

Entidades cargadas:
  1. Opportunity               → OPPORTUNITY
  2. Account                   → ACCOUNT
  3. EstudioCosteBecas__c      → ECBS
  4. BASF_Solicitud__c         → SOLBAN
  5. Case                      → CASES
  6. IndividualEmailResult__c  → EMAIL_RESULTS
  7. CampaignMember            → ACTIVITY_HISTORY
  8. OpportunityFieldHistory   → OPP_FIELD_HISTORY
  9. Pago__c                   → PAGOS
  10. Historial_de_etapa__c    → STAGE_HISTORY

Estado de fases:
  [DONE] Fase 1 – Ingesta SF → Oracle

Uso:
    python src/sf_extract_all.py
    python src/sf_extract_all.py --recreate   # elimina y recrea todas las tablas
"""

import sys
import logging
import logging.handlers
import argparse
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from sf_extractor import SalesforceExtractor
from oracle_connector import OracleConnector

# ---------------------------------------------------------------------------
# Directorios de salida
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
LOG_DIR  = BASE_DIR / "logs"
DOCS_DIR = BASE_DIR / "docs"
LOG_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging con rotación diaria (7 días de retención)
# ---------------------------------------------------------------------------
_root = logging.getLogger()
_root.setLevel(logging.INFO)

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s – %(message)s")

_console = logging.StreamHandler()
_console.setFormatter(_fmt)
_root.addHandler(_console)

_file_handler = logging.handlers.TimedRotatingFileHandler(
    filename=LOG_DIR / "sf_extract_all.log",
    when="midnight",
    backupCount=7,
    encoding="utf-8",
)
_file_handler.setFormatter(_fmt)
_root.addHandler(_file_handler)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filtro base reutilizable (curso + tipos de record)
# ---------------------------------------------------------------------------
CURSO = "2026/2027"
RECORD_TYPES = (
    "('Solicitud Admisión Grado', "
    "'Solicitud Admisión Máster', "
    "'Solicitud Matricula Grado', "
    "'Solicitud Matricula Máster')"
)

# Subquery de Opportunity reutilizada en varios FROM
_OPP_SUBQUERY = (
    f"SELECT id FROM Opportunity "
    f"WHERE PL_Curso_academico__c = '{CURSO}' "
    f"AND RecordType.name IN {RECORD_TYPES}"
)


# ---------------------------------------------------------------------------
# Definición de entidades: (sf_name, oracle_table, pk_col, soql)
# ---------------------------------------------------------------------------
ENTITIES: list[tuple[str, str, str, str]] = [

    # 1 · Opportunity
    ("opportunity", "OPPORTUNITY", "Id", f"""
        SELECT Id, AccountId, TX_Num_Credencial__c, PL_Curso_academico__c,
            PL_Mes_Anio_inicio__c, PL_Tipo_Acceso__c, RecordTypeId, RecordType.name,
            PL_Estado__c, StageName, PL_Subetapa__c, Estado_Matricula__c,
            CH_Simultaneidad__c, CH_Completa__c, Cambia_de_carrera__c,
            PL_Gestionado_por__c, CH_Nacional__c, CH_Internacional__c,
            PL_Origen_de_solicitud__c, PL_Formato__c, CH_Origen_admision__c,
            CloseDate, LeadSource, PL_Origen_formulario__c, PL_Plazo_admision__c,
            LK_Calendario_de_admision__c, LK_Calendario_de_admision__r.name,
            LK_Calendario_de_admision_final__c, LK_Calendario_de_admision_final__r.name,
            ConvocatoriaNombre__c, NU_Nota_media_admision__c, CH_Pruebas_calificadas__c,
            NU_Resultado_admision_puntos__c, CH_StandBy__c, FOR_Estado_tipo_documento__c,
            DT_Fecha_propuesta_resolucion_facultad__c, DT_Nueva_fecha_resolucion_candidato__c,
            NU_Preferencia__c, LK_Titulacion__c, LK_Titulacion__r.name,
            LK_Titulacion_definitiva__c, LK_Titulacion_definitiva__r.name,
            LK_Titulacion_por_curso_academico__c, LK_Titulacion_por_curso_academico__r.name,
            LK_Titulacion_por_curso_academico_final__c, LK_Titulacion_por_curso_academico_final__r.name,
            PL_Resolucion_definitiva__c, PL_Resolucion_inicial__c,
            CH_Titulacion_reorientacion__c, RES_Titulaciones_reorientacion__c,
            CH_Carta_enviada__c, TX_Observaciones_facultad__c, TX_Observaciones_delegado__c,
            ObservacionesAdmision__c, TX_Observaciones_admision__c, Description__c,
            ObservacionesDeCondicionado__c, ObservacionesDeDenegado__c,
            PL_Motivo_cancelacion__c, TX_Motivo_cancelacion__c,
            Observacionesmaster__c, Observacionesresolucion__c, PL_Curso_actual__c,
            LK_CentroEnsenanza__c, LK_CentroEnsenanza__r.name,
            TX_Pais_CentroEnsenanza__c, TX_Provincia_CentroEnsenanza__c,
            TX_Localidad_CentroEnsenanza__c, TX_OtroCentroEnsenanza__c,
            PL_Modalidad_de_acceso__c, PL_Selectividad_UPNA_TrasladoEXP__c,
            CH_Matricula_de_curso_anterior__c, CU_Importe_total__c,
            ImportePrimerPago__c, FechaLimitePrimerPago__c,
            MinimumPaymentPayed__c, MinimumPaymentDate__c,
            Paid_Amount__c, Paid_Percent__c, CH_Pago_superior__c,
            Importe_minimo_personalizado__c, LK_Descuento_Pronto_Pago__c,
            NU_Importe_descuento_pronto_pago__c, NU_Porcentaje_descuento_pronto_pago__c,
            DT_Fecha_creacion_SA__c, DT_Fecha_envio_SA__c,
            DT_Fecha_Publicacion_resolucion__c, DT_Fech_fin_plazo__c,
            TX_Beca_solicitada__c, CH_Matricula_sujeta_beca__c,
            Motivomaster__c, TX_Motivo_solicitud__c,
            PL_Valoracion_estudio_postgrado__c, Fech_Definitiva_Valoraci_n__c,
            TX_Motivo_valoracion__c, ProbabilidadMatriculaDelegado__c,
            DT_Dia_hora_cita__c, CH_Informacion_enviada__c, CH_Conditioned__c,
            CH_Solicita_Alojamiento__c, noContabilizable__c,
            PL_Modalidad_examen_acceso_escogido__c, Kitdevisado__c, FechenvioKit__c,
            DT_Fecha_inicio_Matricula_admision__c, DT_Fecha_fin_Matricula_admision__c,
            DT_Fecha_inicio_Matricula_OOGG__c, DT_Fecha_fin_matricula_OOGG__c,
            NU_PasoMax__c, NU_Paso_solicitud_admision__c,
            CH_Acepto_condiciones_matricula__c, DT_Fecha_Acepta_cond_matr__c,
            PL_Domicilio_durante_curso__c,
            LK_Colegio_M_Residencia_durante_curso__c,
            LK_Colegio_M_Residencia_durante_curso__r.name, OwnerId
        FROM Opportunity
        WHERE PL_Curso_academico__c = '{CURSO}'
        AND RecordType.name IN {RECORD_TYPES}
    """),

    # 2 · Account
    ("account", "ACCOUNT", "Id", f"""
        SELECT Id, ID18__c, ID18__pc, PersonBirthdate, PL_Sexo__pc,
            LK_Nacionalidad__pc, LK_Nacionalidad__pr.name,
            FOR_Pais_de_nacimiento__pc, FOR_Provincia_de_nacimiento__pc,
            TX_Localidad_Nacimiento__pc, PersonMobilePhone,
            CH_Ayuda_financiacion__c, CH_Familia_numerosa__pc,
            PL_Tipo_familia_numerosa__pc, NU_Miembros_familia__pc,
            NU_NumeroHijos__pc, PL_Minusvalia__pc, TX_Grado_minusvalia__pc,
            AccountSource, PersonLeadSource, Rating, PL_Estado__pc,
            PL_Campos_Incompletos__pc, TX_Medio_Origen__c,
            ConvertedFromLead__c, LeadCreatedDate__c,
            GoogleAnalyticsCampaign__pc, GoogleAnalyticsContent__pc,
            GoogleAnalyticsMedium__pc, GoogleAnalyticsSource__pc,
            GoogleAnalyticsTerm__pc, TX_Codigo__pc,
            FO_Pais__pc, FO_Provincia__pc, TX_OtraLocalidad__pc,
            CH_Nacional__pc, CH_Internacional__pc, CH_Grado__pc, CH_Master__pc,
            CH_OtrosTitulos__c, CH_Estudiando_diploma_IB__pc,
            NU_Nota_media_definitiva__pc, PL_Curso_actual__pc, PL_Curso_Interes__c,
            NU_Nota_media_1_bach__pc, BO_Asignacion_nota_media_1_Bach_manual__pc,
            NU_Nota_media_2_bach__pc, BO_Asignacion_nota_media_2_Bach_manual__pc,
            NU_Nota_media_1_bach_SI__pc, NU_Media_Expediente_Universitario__c,
            LK_CentroEnsenanza__pc, LK_CentroEnsenanza__pr.name,
            TX_Pais_CentroEnsenanza__pc, TX_Provincia_CentroEnsenanza__pc,
            TX_Localidad_CentroEnsenanza__pc, TX_OtroCentroEnsenanza__pc,
            NU_Nota_media_2_bach_SI__pc, CH_Estudiante__pc, CH_Solicitante__pc,
            CH_Alumno__pc, CH_AntiguoAlumno__pc, CH_Empleado__pc,
            PL_Lugar_Trabajo_UNAV__pc, CH_Profesor_asociado__pc,
            CH_Hijo_profesor_asociado__pc, CH_Alumni__pc,
            CH_AntiguoAlumno_intercambio__pc, CH_Hijo_empleado__pc,
            CH_Hijo_antiguo_alumno__pc, CH_Hijo_miembro_alumni__pc,
            CH_Hermanos_estudiando_UNAV__pc, CH_Hijo_medico__pc,
            PL_Profesion__pc, TX_Ocupacion__pc, PL_Situacion_socio_economica__pc,
            CH_NoRecibirCorreoPostal__pc, PersonHasOptedOutOfEmail,
            ACC_CHK_Alumno_fidelizado__c, Sudadera__c,
            TX_Observaciones_Backoffice__pc, PL_Cambios_Observ_Automaticas__pc,
            CH_Admision_curso_superior_primero__pc, FOR_Universidad_Afin__pc,
            FOR_Colegio_Afin__pc, CH_Alumno_Colegio_Colaborador__pc,
            CH_Atencion_especial_Admision__pc, CH_Atencion_especial_Rectorado__pc,
            CH_Origen_geografico_estrategico__pc, PL_Aficiones__pc,
            TX_Deporte_practicado__pc, TX_Deporte_federados__pc,
            PL_Conocimiento_UNAV__pc, TX_Otros__pc, PL_Tipo_Centro_Profesor__pc,
            TX_NombreApellidos_Profesor__pc, TX_Delegado_UNAV__pc,
            URL_Web_UNAV__pc, TX_Antiguo_alumno_Grado__pc,
            TX_Antiguo_alumno_Master__pc, TX_Buscador__pc,
            PL_MotivoInfluencia1__pc, PL_MotivoInfluencia2__pc,
            PL_MotivoInfluencia3__pc, TX_Motivo_Influencia__pc,
            ACC_CHK_SeguimientoDelegado__c, ACC_CHK_NoContestaTelefono__c,
            ACC_CHK_NoSeguimientoPersonalizado__c, ACC_TXA_ObservacionesCallCenter__c,
            ACC_DTT_FechaUltimaActividad__c, ACC_CHK_SeguimientoCentro__c,
            ACC_LK_CentroElegidoDelegado__c, ACC_LK_ResponsableSeguimientoCaso__c,
            ACC_CHK_NoSeguimientoCentro__c, ACC_TXA_ObservacionesDelegado__c,
            ACC_DTT_FechaUltimaActividadDelegado__c, ACC_CHK_CasoAsignadoCentro__c,
            ACC_TXA_ObservacionesCentro__c, ACC_DTT_FechaUltimaActividadCentro__c,
            ACC_PL_ValoracionSI__c, PL_valoracionDelegado__c,
            PL_Clasificacion_Inicial__pc, DT_Fech_Clasificacion_Inicial__pc,
            ACC_PL_ProbabilidadMatricula__c, ACC_LK_TitulacionSubjetivaMatricula__c,
            ACC_TXA_ObservacionesProbMatricula__c, ACC_TX_AutorValoracion__c,
            ACC_DTT_FechaValoracion__c, ACC_TXA_ObservacionesAdmision__c,
            ACC_DTT_FechaUltimaActividadAdmision__c, TX_Observaciones_OFE__c,
            DT_Fecha_Observaciones_OFE__c, FechaUltimaActividadOrientacion__c,
            ObservacionesOrientacion__c, ACC_CHK_NoContinua__c, ACC_PL_Motivos__c,
            ACC_TX_GradoMatricula__c, ACC_LK_Universidad__c, ACC_TX_OtraUniversidad__c,
            ACC_TXA_ObservacionesNoContinua__c, ACC_DTT_FechaNoContinua__c,
            CH_Alojamiento__pc, CH_Visitacampus__pc, CH_BecasAyudas__pc,
            CH_Entrevista_personal__pc, DonanteBAN__c, CH_No_Promocionable__c,
            RES_Numero_SI_Grado_Abiertas__c, RES_Numero_SI_Master_Abiertas__c,
            RES_Numero_SA_Grado_Abiertas__c, RES_Numero_SA_Master_Abiertas__c,
            RES_Numero_SM_Grado_Abiertas__c, RES_Numero_SM_Master_Abiertas__c,
            COUNT_SBAN__c, LeadRealInterest__pc, OwnerId
        FROM Account
        WHERE id IN ({_OPP_SUBQUERY.replace('SELECT id', 'SELECT accountid')})
    """),

    # 3 · ECBs (EstudioCosteBecas__c)
    ("ecbs", "ECBS", "Id", f"""
        SELECT Id, LK_oportunidad__c, TX_idEstudio__c, RecordTypeId, RecordType.name,
            LK_titulacionDefinitiva__c, LK_titulacionDefinitiva__r.name,
            LK_titulacionSolicitada__c, LK_titulacionSolicitada__r.name,
            LK_cuenta__c, ECB_LK_SolicitudBAN__c, LK_cursoAcademico__c,
            LK_cursoAcademico__r.name, PL_Estado_oportunidad__c, PL_estado__c,
            FO_regFis_for__c, FO_regFis_ges__c, FO_rentaMEC_for__c,
            FO_rentaFam_ges__c, CU_precioOrdinario_def__c,
            CU_precioIncentivado_def__c, CU_precioFamNum_def__c,
            PO_descFamNum_def__c, CU_precioAplicado_def__c,
            DT_Fecha_solicitud__c, PL_resolucion__c, PL_opcionEscogida__c,
            Plazo_Beca_Alumni__c, PL_estadoBecaAlumni__c, CreatedDate
        FROM EstudioCosteBecas__c
        WHERE LK_oportunidad__c IN ({_OPP_SUBQUERY})
    """),

    # 4 · SOLBAN (BASF_Solicitud__c)
    ("solban", "SOLBAN", "Id", f"""
        SELECT Id, Name, SOL_LK_oportunidad__c, SOL_NUM_curso__c,
            SOL_SEL_estado__c, SOL_LK_titulacionAdmision__c,
            SOL_LK_titulacionAdmision__r.name, SOL_LK_periodoSolicitud__c,
            SOL_LK_periodoSolicitud__r.name, SOL_LK_Solicitante__c,
            RecordTypeId, RecordType.name, SOL_MON_importeConcedido__c,
            CreatedDate
        FROM BASF_Solicitud__c
        WHERE SOL_LK_oportunidad__r.PL_Curso_academico__c = '{CURSO}'
    """),

    # 5 · Cases
    ("cases", "CASES", "Id", f"""
        SELECT Id, AccountId, ContactId, LK_Candidato__c, LK_Oportunidad__c,
            RecordTypeId, RecordType.name, PL_Curso_academico__c,
            Description, Status, PL_Tipo__c, Reason,
            DT_fechaEntrevista__c, DT_fechaEntrevistaAcordada__c,
            PL_temasTratar__c, Subtype__c, CreatedDate,
            LK_Beca__c, Beca_por_curso_academico__c, Modalidad__c,
            PL_Etapa__c, PL_Resolucion__c, CU_ImporteBecado__c,
            NU_Importe_Beca_Concedido__c, NU_Importe_matricula_aproximado__c,
            PO_PorcentajeReduccion__c, PL_Resolucion_beca__c, CH_Pago_integro__c,
            LK_ColegioMayor__c, LK_ColegioMayor__r.name, Origin, Priority,
            OwnerId, PL_Grado_Master__c, URL_AccesoTest__c, CH_URLVisible__c,
            CH_resultadoVisible__c, LK_TitulacionSolicitada__c,
            LK_TitulacionSolicitada__r.name, LK_Titulacion_de_admision__c,
            LK_Titulacion_de_admision__r.name, DT_Fecha_Hora_definitiva__c,
            CH_Archivo_visible__c, DT_fechaOpcion1__c, DT_fechaOpcion2__c,
            NU_asistentes__c, CH_Proceso_de_admision__c, CH_Orientacion__c,
            PL_completarVisita__c, CH_Financiacion__c,
            CursandoBachilleratoInternacional__c, TX_Titulacion_de_interes__c,
            LK_Estudio_de_coste_y_becas__c, PL_Entrevista__c,
            TX_Composicion_familiar__c, NU_Renta_actual__c, TX_Observaciones__c,
            CH_Carta_enviada__c, DT_Fecha_carta_enviada__c, CH_Abonado_alumno__c,
            TX_InformacionBeca__c, TX_MotivoDenegacion__c,
            TX_Motivo_cancelacion_beca__c, BASF_Solicitud__c
        FROM Case
        WHERE PL_Curso_academico__c = '{CURSO}'
        AND LK_Oportunidad__r.RecordType.name IN {RECORD_TYPES}
    """),

    # 6 · IndividualEmailResult
    ("email_results", "EMAIL_RESULTS", "Id", f"""
        SELECT CreatedById, CreatedDate, et4ae5__Contact__c, et4ae5__Lead_ID__c,
            et4ae5__Clicked__c, et4ae5__DateBounced__c, et4ae5__DateOpened__c,
            et4ae5__DateSent__c, et4ae5__DateUnsubscribed__c,
            et4ae5__FromAddress__c, et4ae5__FromName__c, et4ae5__HardBounce__c,
            et4ae5__NumberOfTotalClicks__c, et4ae5__NumberOfUniqueClicks__c,
            et4ae5__Opened__c, et4ae5__SendDefinition__c, et4ae5__SoftBounce__c,
            et4ae5__SubjectLine__c, et4ae5__Tracking_As_Of__c,
            et4ae5__TriggeredSendDefinitionName__c, Id, IsDeleted,
            LastModifiedById, LastModifiedDate, OpportunityId__c, SystemModstamp
        FROM et4ae5__IndividualEmailResult__c
        WHERE OpportunityId__r.PL_Curso_academico__c = '{CURSO}'
        AND OpportunityId__r.RecordType.name IN {RECORD_TYPES}
    """),

    # 7 · ActivityHistory (CampaignMember)
    ("activity_history", "ACTIVITY_HISTORY", "Id", f"""
        SELECT Id, CampaignId, ContactId, LeadId, CreatedDate, Status,
            Estado_del_miembro__c, NU_Acompanantes__c, TX_Acompanantes__c,
            FO_Asiste__c, HasResponded, Centro__c,
            Campaign.LK_tipoActividadPromocion__c,
            Campaign.LK_tipoActividadPromocion__r.name,
            Campaign.Name, Campaign.IsActive, Campaign.PL_tipoAlumno__c,
            Campaign.Status, Campaign.PL_subEtapa__c, Campaign.NUM_Capacidad__c,
            Campaign.AcademicCourse__c, Campaign.EventSurveyType__c,
            Campaign.LK_ColegioRelacionado__c,
            Campaign.LK_ColegioRelacionado__r.name,
            Campaign.PublicoObjetivo__c, Campaign.CH_Online__c,
            Campaign.PL_tipoSolicitud__c, Campaign.TX_lugarEvento__c,
            Campaign.LK_ambitoGeografico__c,
            Campaign.LK_ambitoGeografico__r.name,
            Campaign.FacultadEscuela__c, Campaign.DT_fechaInicio__c,
            Campaign.DT_fechaFin__c, Campaign.Actividad_con_petis__c,
            Campaign.N_mero_de_asistentes_real_o_estimaci_n__c,
            Campaign.N_mero_de_solicitudes_recogidas__c,
            Campaign.Fecha_de_recepci_n_en_Admisi_n__c,
            Campaign.Fecha_fin_de_registro_en_SF__c,
            Campaign.Periodo_de_recepci_n_d_as__c,
            Campaign.Periodo_de_registro_d_as__c,
            Campaign.SUM_AlumnoInscritos__c,
            Campaign.SUM_AcompanantesInscritos__c,
            Campaign.FOR_NUM_Inscritos__c, Campaign.SUM_AlumnosAsistentes__c,
            Campaign.SUM_AcompanantesAsistentes__c, Campaign.FOR_NUM_Asistentes__c,
            Campaign.NumberOfResponses, Campaign.NumberOfLeads,
            Campaign.NumberOfConvertedLeads, Campaign.NumberOfContacts,
            Campaign.NumberOfOpportunities, Campaign.NumberOfWonOpportunities
        FROM CampaignMember
        WHERE ContactId IN (
            SELECT ContactId FROM Opportunity
            WHERE PL_Curso_academico__c = '{CURSO}'
            AND RecordType.name IN {RECORD_TYPES}
        )
    """),

    # 8 · OpportunityFieldHistory
    ("opp_field_history", "OPP_FIELD_HISTORY", "Id", f"""
        SELECT OpportunityId, DataType, NewValue, OldValue, Id, Field
        FROM OpportunityFieldHistory
        WHERE OpportunityId IN ({_OPP_SUBQUERY})
        AND Field IN (
            'PL_Curso_academico__c', 'PL_Tipo_Acceso__c', 'RecordTypeId',
            'PL_Estado__c', 'StageName', 'PL_Subetapa__c', 'CH_Completa__c',
            'Cambia_de_carrera__c', 'PL_Gestionado_por__c', 'CH_Nacional__c',
            'CH_Internacional__c', 'PL_Origen_de_solicitud__c', 'PL_Formato__c',
            'CH_Origen_admision__c', 'CloseDate', 'LeadSource',
            'PL_Origen_formulario__c', 'PL_Plazo_admision__c',
            'LK_Calendario_de_admision__c', 'LK_Calendario_de_admision_final__c',
            'NU_Nota_media_admision__c', 'TX_Descripcion_No_presentado__c',
            'NU_Posicion_ranking__c', 'NU_Resultado_admision_puntos__c',
            'CH_StandBy__c', 'DT_Fecha_propuesta_resolucion_facultad__c',
            'CH_EstudioConvalidaciones_realizado__c', 'LK_Titulacion__c',
            'LK_Titulacion_definitiva__c', 'LK_Titulacion_GA__c',
            'LK_Titulacion_por_curso_academico__c',
            'LK_Titulacion_por_curso_academico_final__c', 'Llamada_Call__c',
            'PL_Resolucion_definitiva__c', 'PL_Resolucion_inicial__c',
            'PL_Resolucion_propuesta__c', 'ObservacionesAdmision__c',
            'ObservacionesDeCondicionado__c', 'ObservacionesDeCondicionadoEng__c',
            'ObservacionesDeDenegado__c', 'ObservacionesDeDenegadoEng__c',
            'Observacionesmaster__c', 'Observacionesresolucion__c',
            'PL_Curso_actual__c', 'LK_CentroEnsenanza__c',
            'TX_Localidad_CentroEnsenanza__c', 'PL_Modalidad_de_acceso__c',
            'CU_Importe_total__c', 'ImportePrimerPago__c',
            'FechaLimitePrimerPago__c', 'MinimumPaymentPayed__c',
            'MinimumPaymentDate__c', 'Paid_Amount__c',
            'NU_Importe_descuento_pronto_pago__c', 'DT_Fecha_creacion_SA__c',
            'DT_Fecha_envio_SA__c', 'DT_Fecha_Publicacion_resolucion__c',
            'DT_Fech_fin_plazo__c', 'TX_Beca_solicitada__c',
            'CH_Matricula_sujeta_beca__c', 'Motivomaster__c',
            'TX_Motivo_solicitud__c', 'PL_Valoracion_estudio_postgrado__c',
            'Fech_Definitiva_Valoraci_n__c', 'TX_Motivo_valoracion__c'
        )
    """),

    # 9 · Pagos
    ("pagos", "PAGOS", "Id", f"""
        SELECT Id, LK_Oportunidad__c, PL_Metodo_pago__c, PL_Tipo__c,
            PL_Origen_pago__c, CU_Importe__c, CH_Pagado__c, LK_Pago__c
        FROM Pago__c
        WHERE LK_oportunidad__c IN ({_OPP_SUBQUERY})
    """),

    # 10 · StageHistory
    ("stage_history", "STAGE_HISTORY", "Id", f"""
        SELECT Id, LK_Oportunidad__c, PL_Etapa__c, PL_Subetapa__c,
            CreatedDate, CH_Completa_principal__c, Fecha_fin_etapa__c
        FROM Historial_de_etapa__c
        WHERE LK_Oportunidad__c IN ({_OPP_SUBQUERY})
    """),
]


# ---------------------------------------------------------------------------
# Generación de documentación Markdown
# ---------------------------------------------------------------------------

def _write_docs(
    results: list[dict],
    recreate: bool,
    start_ts: datetime,
    end_ts: datetime,
) -> Path:
    """
    Genera el fichero docs/FASE1_INGESTA.md con el estado de la ejecución,
    descripción de cada entidad y métricas de carga.
    """
    duration = (end_ts - start_ts).total_seconds()
    ok_rows  = [r for r in results if r["status"] == "OK"]
    err_rows = [r for r in results if r["status"] == "ERROR"]

    total_sf  = sum(r.get("sf_count",  0) for r in ok_rows)
    total_ins = sum(r.get("inserted",  0) for r in ok_rows)
    total_upd = sum(r.get("updated",   0) for r in ok_rows)
    total_db  = sum(r.get("db_after",  0) for r in ok_rows)

    lines: list[str] = []
    lines += [
        "# Fase 1 – Ingesta Salesforce → Oracle",
        "",
        "## ¿Qué hace esta fase?",
        "",
        "Esta fase extrae todas las entidades necesarias de Salesforce mediante la",
        "**REST API SOQL** (paginación automática de 2.000 registros por página),",
        "aplana las relaciones anidadas (`RecordType.name`, `Campaign.Name`, etc.)",
        "y las carga directamente en Oracle **sin generar CSVs intermedios**.",
        "",
        "La carga se realiza mediante **MERGE INTO** (upsert):",
        "- Si el registro **no existe** en Oracle → INSERT.",
        "- Si el registro **ya existe** y algún campo ha cambiado → UPDATE.",
        "- Si el registro ya existe y no ha cambiado → se ignora (sin escritura).",
        "",
        "La inferencia de tipos Oracle (NUMBER, FLOAT, NVARCHAR2, CLOB) se realiza",
        "automáticamente a partir de los valores Python nativos del JSON.",
        "Las columnas CLOB se excluyen de la comparación de cambios (limitación Oracle).",
        "",
        "## Conexión",
        "",
        "| Parámetro | Valor |",
        "|-----------|-------|",
        f"| Host      | `{ENTITIES[0][0]}` ← variable `ORA_HOST` |",
        "| Proxy     | `PMAT_USR[PMATOWNER]` |",
        "| Esquema   | `PMATOWNER` |",
        "| Modo      | oracledb thin (sin cliente Oracle) |",
        "",
        "## Ejecución",
        "",
        f"- **Fecha/hora inicio:** {start_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Fecha/hora fin:**    {end_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Duración total:**    {duration:.1f} s",
        f"- **Modo:**              {'RECREAR tablas (--recreate)' if recreate else 'Upsert incremental'}",
        f"- **Curso académico:**   {CURSO}",
        "",
        "## Resumen global",
        "",
        "| Métrica | Valor |",
        "|---------|-------|",
        f"| Tablas OK    | {len(ok_rows)} / {len(results)} |",
        f"| Tablas ERROR | {len(err_rows)} |",
        f"| Registros SF (total) | {total_sf:,} |",
        f"| Registros en BD (total) | {total_db:,} |",
        f"| Insertados   | {total_ins:,} |",
        f"| Actualizados (aprox.) | {total_upd:,} |",
        "",
        "## Detalle por entidad",
        "",
        "| # | Entidad SF | Tabla Oracle | Registros SF | BD antes | BD después | Insertados | Actualizados | Estado |",
        "|---|-----------|-------------|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for i, r in enumerate(results, 1):
        if r["status"] == "OK":
            lines.append(
                f"| {i} | `{r['sf_name']}` | `{r['ora_table']}` "
                f"| {r['sf_count']:,} | {r['db_before']:,} | {r['db_after']:,} "
                f"| {r['inserted']:,} | {r['updated']:,} | ✅ OK |"
            )
        else:
            lines.append(
                f"| {i} | `{r['sf_name']}` | `{r['ora_table']}` "
                f"| – | – | – | – | – | ❌ ERROR |"
            )

    if err_rows:
        lines += [
            "",
            "## Errores",
            "",
        ]
        for r in err_rows:
            lines.append(f"### `{r['sf_name']}`")
            lines.append(f"```\n{r.get('error', 'Sin detalle')}\n```")
            lines.append("")

    lines += [
        "",
        "## Notas técnicas",
        "",
        "- **Paginación SF:** 2.000 registros por página (límite Salesforce Bulk).",
        "- **Bind variables:** los nombres con caracteres especiales (`.`, `__r`) se",
        "  sanitizan reemplazando caracteres no alfanuméricos por `_`.",
        "- **CLOB en change_cond:** Oracle no permite comparar columnas CLOB con `!=`.",
        "  Estas columnas se actualizan siempre que la fila haya coincidido.",
        "- **Conteo de actualizados:** aproximado. Se calcula como",
        "  `registros_SF – insertados`. Los registros coincidentes pero sin cambios",
        "  no generan UPDATE y podrían inflar esta cifra.",
        f"- **Log completo:** `logs/sf_extract_all.log`",
        "",
        "---",
        f"*Generado automáticamente el {end_ts.strftime('%Y-%m-%d %H:%M')} UTC*",
    ]

    doc_path = DOCS_DIR / "FASE1_INGESTA.md"
    doc_path.write_text("\n".join(lines), encoding="utf-8")
    return doc_path


# ---------------------------------------------------------------------------
# Ejecución principal: SF → Oracle directo
# ---------------------------------------------------------------------------

def run(recreate: bool = False) -> list[dict]:
    """
    Ejecuta la ingesta completa: extrae cada entidad de Salesforce y la
    carga directamente en Oracle via upsert (MERGE INTO).

    Args:
        recreate: si True, elimina y recrea cada tabla antes de cargar.

    Returns:
        Lista de dicts con métricas por entidad.
    """
    start_ts = datetime.now(timezone.utc)

    logger.info("=" * 70)
    logger.info("FASE 1 – INGESTA SALESFORCE → ORACLE  [INICIO]")
    logger.info("Curso académico: %s | Modo: %s",
                CURSO, "RECREAR tablas" if recreate else "upsert incremental")
    logger.info("=" * 70)

    sf = SalesforceExtractor()
    sf.authenticate()

    results: list[dict] = []

    with OracleConnector() as ora:
        for sf_name, ora_table, pk_col, soql in ENTITIES:
            logger.info("-" * 60)
            logger.info("▶ Procesando: %s → %s", sf_name, ora_table)
            row: dict = {"sf_name": sf_name, "ora_table": ora_table}

            try:
                # ── 1. Extraer de Salesforce ─────────────────────────────
                records_raw = sf.query(soql)
                logger.info("  SF: %d registros obtenidos", len(records_raw))

                if not records_raw:
                    logger.warning("  Sin registros en SF para %s.", sf_name)
                    row.update({"status": "OK", "sf_count": 0,
                                "db_before": 0, "db_after": 0,
                                "inserted": 0, "updated": 0})
                    results.append(row)
                    continue

                # ── 2. Aplanar relaciones anidadas ───────────────────────
                flat_records = [
                    SalesforceExtractor._flatten_record(r)
                    for r in records_raw
                ]

                # ── 3. Recrear tabla si se pide ──────────────────────────
                if recreate:
                    ora.drop_table_if_exists(ora_table)

                # ── 4. Upsert en Oracle ──────────────────────────────────
                metrics = ora.upsert_records(flat_records, ora_table, pk_col)

                row.update({"status": "OK", **metrics})
                results.append(row)

                logger.info(
                    "  ✔ %-20s | SF: %6d | BD antes: %6d | "
                    "BD después: %6d | Insertados: %6d | Actualizados: ~%6d",
                    ora_table,
                    metrics["sf_count"],
                    metrics["db_before"],
                    metrics["db_after"],
                    metrics["inserted"],
                    metrics["updated"],
                )

            except Exception as exc:
                logger.error("  ✖ ERROR en '%s': %s", sf_name, exc, exc_info=True)
                row.update({"status": "ERROR", "error": str(exc)})
                results.append(row)

    end_ts = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Resumen en consola
    # ------------------------------------------------------------------
    ok_rows  = [r for r in results if r["status"] == "OK"]
    err_rows = [r for r in results if r["status"] == "ERROR"]

    logger.info("=" * 70)
    logger.info("FASE 1 – INGESTA SALESFORCE → ORACLE  [FIN]")
    logger.info("=" * 70)
    logger.info("")
    logger.info("%-22s %8s %10s %10s %10s %10s",
                "Entidad", "SF", "BD antes", "BD después", "Insertados", "Actualizados")
    logger.info("-" * 80)
    for r in results:
        if r["status"] == "OK":
            logger.info(
                "%-22s %8d %10d %10d %10d %10d",
                r["sf_name"], r["sf_count"], r["db_before"],
                r["db_after"], r["inserted"], r["updated"],
            )
        else:
            logger.error("%-22s  ERROR: %s", r["sf_name"], r.get("error", "")[:60])
    logger.info("-" * 80)
    logger.info(
        "%-22s %8d %10s %10d %10d %10d",
        "TOTAL",
        sum(r.get("sf_count",  0) for r in ok_rows),
        "",
        sum(r.get("db_after",  0) for r in ok_rows),
        sum(r.get("inserted",  0) for r in ok_rows),
        sum(r.get("updated",   0) for r in ok_rows),
    )
    logger.info("")
    if err_rows:
        logger.warning("Fase 1 completada con %d errores.", len(err_rows))
    else:
        logger.info("Fase 1 completada sin errores.")

    # ------------------------------------------------------------------
    # Documentación Markdown
    # ------------------------------------------------------------------
    doc_path = _write_docs(results, recreate, start_ts, end_ts)
    logger.info("Documentación generada: %s", doc_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fase 1: Ingesta Salesforce → Oracle"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Elimina y recrea todas las tablas Oracle antes de cargar (útil en setup inicial)",
    )
    args = parser.parse_args()
    run(recreate=args.recreate)
