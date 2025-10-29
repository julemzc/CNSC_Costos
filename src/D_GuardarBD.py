#!/usr/bin/env python
# coding: utf-8

# Librerias generales
import numpy as np
import socket
import json
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Funciones Generales
from src.A_Generales import lprint, closeConnection, fConsultaScript, fEjecutaScript, fRetornaLista, openCosteo, fInsertaRegistros, fCorregirInt # type: ignore
from src.B_Historico import fVariables # type: ignore

# Generar el Json de auditoria
def fEjecucionJson(Dict):
    # campo np.ejecucion.variables
    variables = json.dumps(fVariables().to_dict(orient='records'), indent=1, ensure_ascii=False)

    # campo np.ejecucion.redNeuronal
    escena = list(Dict.keys())[0]
    rn_list = []
    for i, hi in enumerate(Dict[escena]['Historico']):
        red = hi['red']
        # Agregar un diccionario con los valores deseados
        rn_list.append({
            "id": red[0],
            "neuronas": red[1],
            "nombre": str(red[6]).strip(),
            "learnrate": red[2],
            "dropout": red[3],
            "epocas": red[4],
            "batch": red[5]
        })
    redNeuronal = json.dumps(rn_list, indent=1, ensure_ascii=False)

    # campo np.ejecucion.escenarios
    ml = openCosteo()[1]
    query = f"""
    SELECT es.id escena, de.id, rank() OVER (PARTITION BY es.id ORDER BY de.id) AS ord,
    es.nombre, campo, operador, valor, percmin, percmax, coalesce(concatena,'') concatena, divide_ascenso
    FROM {ml}.np_escenas es
    JOIN {ml}.np_escenas_detalle de on (es.id = de.escena_id)
    WHERE es.activo AND de.activo
    ORDER BY es.id, de.id
    """
    dfEscenarios = fConsultaScript(openCosteo, query)
    escenarios = json.dumps(dfEscenarios.to_dict(orient='records'), indent=1, ensure_ascii=False)
    return variables, redNeuronal, escenarios


# #### Guardar resultados en las tablas asociadas

# Insertar datos en nn_ejecucion
def fInsertar_npEjecucion(Dict, dfCostos, ml, filtro=None):
    variables, redNeuronal, escenarios = fEjecucionJson(Dict)

    filtro = filtro.replace("'", "").strip()
    filtro = f"Excel: {filtro}" if filtro.endswith(".xlsx") else f"convocatoria_id in ({filtro})"
  
    #### Crea una nueva ejecución
    query = f"""
    INSERT INTO {ml}.np_ejecucion (id, fecha_creacion, hostname, ip, variables, redneuronal, escenarios, filtro)
    SELECT COALESCE(max(id),0) + 1 AS maximo,
    CURRENT_TIMESTAMP as fecha_creacion,
    '{socket.gethostname()}' AS hostname,
    '{socket.gethostbyname(socket.gethostname())}' AS ip,
    '{variables}' AS variables,
    '{redNeuronal}' AS redneuronal,
    '{escenarios}' AS escenarios,
    '{filtro}' AS filtro
    FROM {ml}.np_ejecucion
    """
    fEjecutaScript(openCosteo, query)

    #### Calcula ultima ejecución y la hora del sistema
    query = f"""
        SELECT COALESCE(max(id),0) AS maximo, CURRENT_TIMESTAMP AS tiempo
        FROM {ml}.np_ejecucion
    """
    dfTemp = fConsultaScript(openCosteo, query)

    ejecucion = dfTemp.iloc[0]['maximo']
    fecha_creacion = dfTemp.iloc[0]['tiempo']

    #### Agregar campos adicionales a dfCostos
    dfCostos['id'] = dfCostos.index
    dfCostos['ejecucion_id'] = ejecucion
    dfCostos['fecha_creacion'] = fecha_creacion

    lprint(f"Ejecucion creada: {str(ejecucion)}")
    return dfCostos, ejecucion


# Insertar datos en nn_empleo
def fInsertar_nnEmpleo(dfCostos):
    #### retira las columnas que no son de la tabla
    conx, ml = openCosteo()
    query = f"""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = '{ml}'
    AND table_name = 'nn_empleo';
    """
    columnas = fRetornaLista(openCosteo, query)

    setA = set(dfCostos.columns.values)
    setB = set([item.replace("'", "") for item in columnas.split(',')])
    list_diff = list(setA - setB)
    dfCostos = dfCostos.drop(columns=list_diff)

    #### Insertar las OPEC en la tabla nn_empleo
    fInsertaRegistros(conx, ml, 'nn_empleo', dfCostos)
    return dfCostos


#  Insertar datos en nn_stats
def fInsertar_nnStats(Dict, dfCostos):
    lprint("Inicio insertar nn_stats")
    ## Conexión a los datos de Prueba
    conexion, ml = openCosteo()
    cursor = conexion.cursor()

    #### Insertar en la tabla nn_stats
    sql = f"""
    INSERT INTO {ml}.nn_stats
    (ejecucion_id, escenario_id, redneuronal_id, r2, mse, rmse, mae, porc_rm, porc_escritas, fecha_creacion, ascenso)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    for e in Dict:
        boolAscenso = bool(Dict[e].get('Ascenso', False))

        for hist in Dict[e]['Historico']:
            pred_scl = hist['model'].predict(Dict[e]['X'])
            pred_scl = np.asarray(pred_scl).reshape(-1, 1)
            pred_inv = Dict[e]['scaler_Y'].inverse_transform(pred_scl)
            pred_y = np.expm1(pred_inv).flatten()

            real_inv = Dict[e]['scaler_Y'].inverse_transform(Dict[e]['Y'].reshape(-1,1))
            real_y = np.expm1(real_inv).flatten()

            assert pred_y.shape == real_y.shape
#            assert np.all(np.isfinite(pred_y)) and np.all(np.isfinite(real_y))
            pred_y = np.clip(pred_y, max(real_y.min(), 1), real_y.max()*1.5)

            r2 = r2_score(real_y, pred_y)
            mse = mean_squared_error(real_y, pred_y)
            rmse = sqrt(mse)
            mae = mean_absolute_error(real_y, pred_y)

            lprint(str(dfCostos['ejecucion_id'][0]) +"/"+ str(Dict[e]['Id'])+"/"+ str(hist['red'][0])+"/"+ 
                   str(r2)+"/"+str(mse)+"/"+ str(rmse)+"/"+ str(mae)+"/"+
                   str(Dict[e]['Porc_RM'])+"/"+ str(Dict[e]['Porc_Escrit'])+"/"+ str(dfCostos['fecha_creacion'][0])+"/"+ str(boolAscenso))
            
            cursor.execute(sql, (str(dfCostos['ejecucion_id'][0]), str(Dict[e]['Id']), hist['red'][0], 
                                 float(r2), float(mse), float(rmse), float(mae),
                                 float(Dict[e]['Porc_RM']), float(Dict[e]['Porc_Escrit']), 
                                 dfCostos['fecha_creacion'][0], boolAscenso))

    conexion.commit()
    closeConnection(conexion)
    cursor.close()
    lprint("Fin insertar nn_stats")


# Insertar datos en nn_proyeccion
def fInsertar_nnProyeccion(Dict, dfCostos, Convocatoria):
    lprint("Inicio insertar nn_proyeccion")
    ## Conexión a los datos de Prueba
    conexion, ml = openCosteo()
    cursor = conexion.cursor()

    #### Calcula último id de proyeccion
    query = f"""
    select COALESCE(max(id),0) as id 
    from {ml}.nn_proyeccion
    """
    dfTemp = fConsultaScript(openCosteo, query)
    ProyeccionId = int(dfTemp.iloc[0]['id'])

    #### Insertar en la tabla 'nn_proyeccion' las proyecciones por cada empleo, escenario y red neuronal
    sqlProy = f"""
    INSERT INTO {ml}.nn_proyeccion
    (id, ejecucion_id, escenario_id, redneuronal_id, nn_empleo_id, mun_inscritos, mun_aprobo_vrm, mun_aprobo_escritas, fecha_creacion, ascenso)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    for empleo in Convocatoria:
        for e in Dict:
            f = Dict[e]['Filtro']
            empleo[f]['ProyeccionId'] = []
            divAsc = Dict[e]['Ascenso']
            empAsc = empleo.get('concurso_ascenso')
            if divAsc is None or empAsc is None or divAsc == empAsc:
                for hist in Dict[e]['Historico']:
                    inscritos = empleo[f]['Neuronas'][hist['red'][0]]
                    inscritosRM = int(round(Dict[e]['Porc_RM'] * inscritos,0))
                    inscritosEsc = int(round(Dict[e]['Porc_Escrit'] * inscritos,0))
                    ProyeccionId += 1
                    empleo[f]['ProyeccionId'].append(ProyeccionId)
                    cursor.execute(sqlProy, (ProyeccionId, str(dfCostos['ejecucion_id'][0]), str(Dict[e]['Id']), 
                                             hist['red'][0], str(empleo['Indice']), str(inscritos), str(inscritosRM), 
                                             str(inscritosEsc), dfCostos['fecha_creacion'][0], bool(divAsc)))

    conexion.commit()
    closeConnection(conexion)
    cursor.close()
    lprint("Fin insertar nn_proyeccion")


# Selección de la mejor opción y actualización de resultados de inscritos
def fRankProyeccion(ml, ejecucion):
    lprint("Inicio Ranking proyeccion")
    scaler = MinMaxScaler()
    dfStats = fConsultaScript(openCosteo, f"select * from {ml}.nn_stats where ejecucion_id = {str(ejecucion)}")
    if dfStats.shape[0] > 1:
      dfStats[['r2_norm', 'rmse_norm', 'mae_norm']] = scaler.fit_transform(dfStats[['r2', 'rmse', 'mae']])

    weight_r2, weight_rmse, weight_mae = [0.5, 0.25, 0.25]    # opción 2
    
    dfStats['score'] = weight_r2 * dfStats['r2_norm'] - weight_rmse * dfStats['rmse_norm'] - weight_mae * dfStats['mae_norm']
    dfStats['rank'] = fCorregirInt(dfStats['score'].rank(ascending=False))
    
    for i, row in dfStats.iterrows():
        queryUpdate = f"UPDATE {ml}.nn_stats SET ranking = {row['rank']} WHERE ejecucion_id = {str(ejecucion)} AND escenario_id = {row['escenario_id']} AND redneuronal_id = {row['redneuronal_id']} AND ascenso = {row['ascenso']}"
        fEjecutaScript(openCosteo,queryUpdate)
    lprint("Fin Ajustar Rank de las proyecciones")


def fUpdateEmpleo(ml, ejecucion):
    ### Actualiza la elegida
    queryUpdate = f"""
    UPDATE {ml}.nn_stats ns
    SET elegida = TRUE
    FROM (
        SELECT escenario_id, redneuronal_id, ejecucion_id
        FROM {ml}.nn_stats ns
        WHERE ns.ejecucion_id = {str(ejecucion)} AND NOT ascenso
        ORDER BY ranking LIMIT 1
    ) rk
    WHERE ns.escenario_id = rk.escenario_id AND ns.redneuronal_id = rk.redneuronal_id AND ns.ejecucion_id = rk.ejecucion_id
    """
    fEjecutaScript(openCosteo,queryUpdate)

    #### Actualiza la cantidad de inscritos en la tabla nn_empleo
    queryUpdate = f"""
    UPDATE {ml}.nn_empleo SET
    mun_inscritos = pr.mun_inscritos,
    mun_aprobo_vrm = pr.mun_aprobo_vrm,
    mun_aprobo_escritas = pr.mun_aprobo_escritas
    FROM (
        SELECT np.ejecucion_id, nn_empleo_id, mun_inscritos, mun_aprobo_vrm, mun_aprobo_escritas
        FROM {ml}.nn_stats ns
        NATURAL JOIN {ml}.nn_proyeccion np
        WHERE elegida AND ns.ejecucion_id = {str(ejecucion)}
    ) AS pr
    WHERE nn_empleo.id = pr.nn_empleo_id
    AND nn_empleo.ejecucion_id = pr.ejecucion_id
      """
    fEjecutaScript(openCosteo,queryUpdate)
    lprint("Fin Actualizar mejor Proyección")


# Actualiza que se ha ejecutado
def fProyectaConvocatoria(ml, ejecucion, listaConv):
  query = f"""
      UPDATE {ml}.seleccion_convocatoria
      SET ejecucion_id = {str(ejecucion)}, proyectar = True, fecha_ejecucion = CURRENT_TIMESTAMP
      WHERE id in ({str(listaConv)})"""
  fEjecutaScript(openCosteo, query)


# Guardar resultados en la Base de Datos
def fGuardarResultados(Dict, dfCostos, Convocatoria, co):
    lprint("Inicio - Guardar resultados en la BD")
    ml = openCosteo()[1]
    dfCostos, ejecucion = fInsertar_npEjecucion(Dict, dfCostos, ml, co[0])

    if dfCostos.shape[0] > 0:
        #Insertar los datos en cada una de las tablas de bdCosteo
        dfCostos = fInsertar_nnEmpleo(dfCostos)
        fInsertar_nnStats(Dict, dfCostos)
        fInsertar_nnProyeccion(Dict, dfCostos, Convocatoria)
        #Seleccionar la proyección adecuada para la convocatoria a proyectar
        fRankProyeccion(ml, ejecucion)
        fUpdateEmpleo(ml, ejecucion)
        fProyectaConvocatoria(ml, ejecucion, co[1])
        lprint('Inserción en tablas finalizada')
    lprint("FIN - Resultados guardados en la BD \n")
    return dfCostos, ejecucion
