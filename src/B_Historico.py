#!/usr/bin/env python
# coding: utf-8

# #### Librerias
import pandas as pd
import math
import pickle
from sqlalchemy.types import String

# Funciones Generales
from src.A_Generales import lprint, openSimo, openCosteo, fConsultaScript, fConsultaTabla, fCrearTabla, engineCosteo, fSalarios, fRetornaLista, fCorregirInt, rTecho, ConsultaSQL, fEjecutaDDL, fEjecutaScript # type: ignore

#Retorna los datos de la tabla nn_simo
def rDatosSimo():
    lprint("Inicio - Crear y Leer historico de Simo")
    if nnSimoUltimo():
        lprint("\n\nConsulta Simo y completar base nn_simo")
        df = ConsultaSQL(openSimo, 'sql/historico.sql')

        df = pd.merge(df, bdMunicipio(), on='codigo_dane', how='left')
        df = pd.merge(df, ConsultaSQL(openSimo, 'sql/experiencia.sql'), on='empleo_id', how= 'left')
        df = fRellenarDatos(df)
        df = fAgregarSmmlvAgno(df)

        # corregir tipo de datos
        df, dty = fCorregirTipoDatos(df)

        # Insertar datos histórico en la tabla nn_simo
        ml = openCosteo()[1]
        fEjecutaScript(openCosteo, f'DROP VIEW IF EXISTS {ml}.nn_simo_unico;')
        fCrearTabla(df, engineCosteo(), ml,'nn_simo', dty)
        fEjecutaDDL(openCosteo, 'sql/nn_simo_unico.sql')

        # consulta nn_simo
        lprint("Consulta tabla nn_simo")
        df = fConsultaSimo()

        with open('config/dfEmpleo.pkl', 'wb') as fileP:
            pickle.dump(df, fileP)
    else:
        with open('config/dfEmpleo.pkl', 'rb') as fileP:
            df = pickle.load(fileP)
    lprint("FIN de lectura de nn_simo \n")
    return df


# retorna si debe actualizar la tabla nn_simo si supera los días del parametro
def nnSimoUltimo():
    ml = openCosteo()[1]
    dias = int(fConsultaScript(openCosteo, f"select valor from {ml}.np_parametros where tipo = 'nn_simo'").loc[0,'valor'])

    query =  f"""
        select max(fecha_actualizacion) fecha_simo, 
        current_timestamp AT TIME ZONE 'EST' AS fecha_actual
        from {ml}.nn_simo
        """
    df = fConsultaScript(openCosteo, query)
    df['diferencia'] = df['fecha_actual'].dt.tz_localize(None) - df['fecha_simo'].dt.tz_localize(None)
    df['diferencia'] = df['diferencia'].dt.total_seconds() / (60 * 60 * 24)
    diferencia = math.ceil(df['diferencia'].loc[0])
    return True if diferencia > dias else False


# Corregir datos del salario y del nivel
def fRellenarDatos(df):
    lprint("Inicia fCompletarDatos()")

    # Tablas de corrección
    grado_salario = fConsultaTabla(openCosteo, 'grado_salario')
    grado_nivel = fConsultaTabla(openCosteo, 'grado_nivel')

    # 1. Actualiza el año en base al grado, nivel y salario
    vs = df.merge(grado_salario, left_on=['grado_nivel_id', 'asignacion_salarial'], right_on=['grado_nivel', 'salario'], how='inner')
    vs0 = vs[fCorregirInt(vs['vigencia_salarial']) == 0]
    if vs0.shape[0] > 0:
        df.loc[df['empleo_id'].isin(vs0['empleo_id']), 'vigencia_salarial'] = vs0['agno'].values

    # 2. Actualiza el año y el grado en base al nivel y salario
    vs = df.merge(grado_salario, left_on='asignacion_salarial', right_on='salario')
    vs = vs.merge(grado_nivel, left_on='grado_nivel', right_on='id', suffixes=('', '_gn1'))
    vs = vs.merge(grado_nivel, left_on='grado_nivel_id', right_on='id', suffixes=('', '_gn2'))
    
    vs0 = vs[(vs['nivel_id'] == vs['nivel_id_gn2']) & (fCorregirInt(vs['vigencia_salarial']) == 0) & (vs['grado_nivel'] != vs['id_gn2'])]
    if vs0.shape[0] > 0:
        df.loc[df['empleo_id'].isin(vs0['empleo_id']), 'vigencia_salarial'] = vs0['agno'].values
        df.loc[df['empleo_id'].isin(vs0['empleo_id']), 'grado_nivel_id'] = vs0['grado_nivel'].values
        df.loc[df['empleo_id'].isin(vs0['empleo_id']), 'grado'] = vs0['grado_gn1'].values

    # 3. Actualiza el grado y el nivel en base al salario y el año
    vs = df.merge(grado_salario, left_on=['vigencia_salarial', 'asignacion_salarial'], right_on=['agno', 'salario'])
    vs = vs.merge(grado_nivel, left_on='grado_nivel', right_on='id', suffixes=('', '_gn1'))
    vsn = vs[vs['grado_nivel_id'].isnull()]
    vsc = vsn.groupby('empleo_id').size().reset_index(name='cont')
    vs1 = vsc[vsc['cont'] == 1]
    vs_final = vsn[vsn['empleo_id'].isin(vs1['empleo_id'])]
    if vs_final.shape[0] > 0:
        df.loc[df['empleo_id'].isin(vs_final['empleo_id']), 'grado_nivel_id'] = vs_final['grado_nivel'].values
        df.loc[df['empleo_id'].isin(vs_final['empleo_id']), 'grado'] = vs_final['grado_gn1'].values
    return df


# Agregar campos faltantes
def fAgregarSmmlvAgno(df):
    # Ajustes a campos
    df['agno'] = df['vigencia_salarial'].fillna(df['conv_agno'])
    df['experiencia'] = df['experiencia'].str[:999]

    # Ajuste de Salarios mínimos
    sl = pd.merge(df, fSalarios(), on='agno', how='inner')
    sl['smmlv'] = round(sl['asignacion_salarial'] / sl['agno_smmlv'],2)
    sl1 = sl[fCorregirInt(sl['smmlv']) >= 1]
    if sl1.shape[0] > 0:
        df.loc[df['empleo_id'].isin(sl1['empleo_id']), 'smmlv'] = sl1['smmlv'].values
    else:
        df['smmlv'] = 0
    return df


# Corregir Tipo de Datos
def fCorregirTipoDatos(df):
    # ajustar datos Int permitiendo NULL
    for col in ['vigencia_salarial', 'asignacion_salarial', 'grado_nivel_id', 'nivelid', 'grado', 'conv_id', 'conv_agno', 'conv_padre_id', 'criterio_id', 'agno']:
        df[col] = fCorregirInt(df[col],True)
    
    # ajustar datos Int ajustando a 0
    for col in ['empleo_id', 'vacantes_opec', 'vacantes_municipios', 'vacantes', 'exp_profesional', 'exp_prof_relacionada', 'exp_laboral', 'exp_labo_relacionada', 'exp_relacionada']:
        df[col] = fCorregirInt(df[col])

    # ajustar datos de la base nn_simo
    if 'inscritos' in df.columns:
        for col in ['inscritos', 'aprobo_vrm', 'aprobo_escritas', 'mun_inscritos', 'mun_aprobo_vrm', 'mun_aprobo_escritas']:
            df[col] = fCorregirInt(df[col])

    for col in ['concurso_ascenso','etiqueta','sin_experiencia']:
        df[col] = df[col].fillna(False).astype('bool')

    # Aplicar longitud a los campos text
    dty = {}
    for col, dtype in df.dtypes.items():
        if dtype == 'object':
            largo = rTecho(df[col].str.len().max()) if df[col].notna().any() else 1
            dty[col] = String(largo)
    return df, dty


# #### Ajuste de datos del smmlv y selección de columnas
def fConsultaSimo():
    lprint("Inicia fConsultaSimo()")

    lista = fRetornaLista(openCosteo, f"""SELECT nombre, tipo, id FROM {openCosteo()[1]}.np_variables WHERE activo """)
    lista = [x.strip("'") for x in lista[0].split(',')]

    col_mun = fRetornaLista(openCosteo, f"""SELECT nombre FROM {openCosteo()[1]}.np_variables WHERE no_unico """)
    col_mun = [x.strip("'") for x in col_mun.split(',')]
    col_mun = col_mun + ['mun_inscritos', 'mun_aprobo_vrm', 'mun_aprobo_escritas']

    if any(item in lista for item in col_mun):
        df = fConsultaTabla(openCosteo, 'nn_simo')
        lprint('nn_simo')
    else:
        df = fConsultaTabla(openCosteo, 'nn_simo_unico')
        for col in col_mun:
            df[col] = None
        df['vacantes'] = df['vacantes_opec']
        df['mun_inscritos'] = df['inscritos']
        df['mun_aprobo_vrm'] = df['aprobo_vrm']
        df['mun_aprobo_escritas'] = df['aprobo_escritas']
        lprint('nn_simo_unico')

    df = fCorregirTipoDatos(df)[0]

    if any(item in lista for item in ['smmlv', 'asignacion_salarial']):
        df = df[~pd.isnull(df['smmlv'])]
        df.reset_index(drop=True, inplace=True)

    lprint(f"Tabla empleo {str(df.shape)} Fin fConsultaSimo()")
    return df


# ####  Consulta de Empleos a proyectar 
def rConvocatoriaSimo(id_convocatoria =0):
    lprint("Inicio - Seleccionar convocatoria a proyectar")
    df = pd.DataFrame({})
    ml = openCosteo()[1]

    lprint(f'La convocatoria es: {str(id_convocatoria)}')
    if id_convocatoria == 0:
        id_convocatoria = fConsultaScript(openCosteo, f"select valor from {ml}.np_parametros where tipo = 'convocatorias'").loc[0,'valor']
    query = f"""
    SELECT convocatoria_id, id
    FROM seleccion_convocatoria
    WHERE NOT proyectar
    AND ejecucion_id IS NULL
    AND convocatoria_id IN ({str(id_convocatoria)})
    ORDER BY id
    LIMIT 1
    """
    conv = fRetornaLista(openCosteo, query)

    if conv != None:
        lprint(f'Consulta de la convocatoria {str(conv[0])}')

        lista = fRetornaLista(openCosteo, f"""SELECT nombre, tipo, id FROM {openCosteo()[1]}.np_variables WHERE activo """)
        lista = [x.strip("'") for x in lista[0].split(',')]

        col_mun = fRetornaLista(openCosteo, f"""SELECT nombre FROM {openCosteo()[1]}.np_variables WHERE no_unico """)
        col_mun = [x.strip("'") for x in col_mun.split(',')]
        col_mun = col_mun + ['mun_inscritos', 'mun_aprobo_vrm', 'mun_aprobo_escritas']

        # si hay una variable de municipio
        if any(item in lista for item in col_mun):
            df = ConsultaSQL(openSimo, 'sql/convocatorias.sql', conv[0])
        else:
            df = ConsultaSQL(openSimo, 'sql/convocatorias_unico.sql', conv[0])

        # Capturar la categoria del municipio
        df = pd.merge(df, bdMunicipio(), on='codigo_dane', how='left')
        df = pd.merge(df, ConsultaSQL(openSimo, 'sql/experiencia.sql'), on='empleo_id', how='left')
        lprint('Cruce con los municipios y la experiencia')
        df['etiqueta'] = df['etiqueta'].fillna(False).astype('bool')
        df = fAgregarSmmlvAgno(df)
        df, dty = fCorregirTipoDatos(df)
        lprint(f"FIN de consulta de la convocatoria {str(df.shape)}")
    else:
        lprint("Convocatoria NO se encuentra seleccionada")
    lprint("FIN de proyección de convocatoria")
    return df, conv


# Si la convocatoria se recibe en un Excel
def bdConvocatoriaExcel(dfExcel):
    dfConvLista = dfExcel['empleo_id'].tolist()

    esquema = openSimo()[1]
    script = f"""
    SELECT
    emp.id empleo_id,
    emp.grado_nivel_id,
    grado_nivel.grado,
    emp.asignacion_salarial,
    ent.nit,
    ent.nombre entidad,
    te.nombre tipo_entidad,
    mun.cod_dane,
    mun.nombre municipio,
    emp.vigencia_salarial agno,
    nivel.nombre as nivel,
    deno.nombre deno_nombre
    FROM {esquema}.empleo emp
    LEFT JOIN {esquema}.entidad ent ON (ent.id=emp.entidad_id)
    LEFT JOIN {esquema}.tipo_entidad te ON (ent.tipo_entidad_id = te.id)
    LEFT JOIN {esquema}.municipio mun ON (ent.municipio_id = mun.id)
    LEFT JOIN {esquema}.departamento dept ON (dept.id=mun.departamento_id)
    LEFT JOIN {esquema}.grado_nivel ON (emp.grado_nivel_id = grado_nivel.id)
    LEFT JOIN {esquema}.nivel ON (grado_nivel.nivel_id = nivel.id)
    LEFT JOIN {esquema}.denominacion deno ON (emp.denominacion_id=deno.id)
    WHERE emp.id IN ({', '.join(map(str, dfConvLista))});
    """
    dfSimo = fConsultaScript(openSimo,script)

    dfSimo = pd.merge(dfSimo, bdMunicipio(), on='codigo_dane', how='left')
    dfSimo = pd.merge(dfSimo, ConsultaSQL(openSimo, 'sql/experiencia.sql'), on='empleo_id', how= 'inner')

    dfCostos = pd.merge(dfExcel,dfSimo,on=['empleo_id'],how='inner')
    return dfCostos


# ####  Consulta al municipio en fcd2
def bdMunicipio():
    ml = openCosteo()[1]
    query = f"""
    SELECT codigo codigo_dane, municipio_categoria_id::varchar mun_categoria
    FROM {ml}.lugar
    WHERE tipo_lugar_id = 3
    """
    dfM = fConsultaScript(openCosteo, query)
    return dfM.dropna()


# Consulta las variables
def fVariables():
    ml = openCosteo()[1]
    query = f"""
    SELECT nombre, tipo, id
    FROM {ml}.np_variables
    WHERE activo
    """
    return fConsultaScript(openCosteo, query)
