#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pandas.io.sql as psql
import psycopg2
import yaml
import re
import math
import logging
from psycopg2 import sql
from sqlalchemy import create_engine
from datetime import datetime
from unidecode import unidecode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_file='config/config.yaml'):
    with open(config_file, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# log de ejecución
def lprint(mensaje, log_dir="./logs/"):
    log_file = log_dir + "log_" + datetime.now().strftime('%Y%m%d') + ".txt"
    log = datetime.now().strftime('%Y-%h-%d-%H:%M:%S') + ', ' + mensaje + '\n'
#    print(log)
    logger.info(log)
    with open(log_file, "a") as f:
        f.write(log)

# #### Abrir y cerrar conexión
def openConnection(conn_name):
    config = load_config().get(conn_name, {})
    try:
#        lprint(f"Conectando a: {config.get('host', 'host no especificado')}")
        conn = psycopg2.connect(
            user=config['user'],
            password=config['pass'],
            host=config['host'],
            port=config['port'],
            database=config['db']
        )
        return conn, config['schema']
    except psycopg2.Error as e:
        lprint(f" Error al conectar con PostgreSQL ({conn_name}): {e}")
        return None, None

def closeConnection(conexion):
    try:
        if(conexion):
            conexion.close()
    except Exception as e:
        lprint(str(e))


# #### Parametros de conexión a cada base de datos
def openSimo():
    return openConnection('conn_chami')

def openWayu():
    return openConnection('conn_wayu')

def openCosteo():
    return openConnection('conn_costeo')


# #### Parametros de acciones DML
def openEngine(conn_name):
    config = load_config().get(conn_name, {})
    try:
        conn_str = f"postgresql+psycopg2://{config['user']}:{config['pass']}@{config['host']}:{config['port']}/{config['db']}"
        lprint(f"Creando engine para: {config['host']}")
        return create_engine(conn_str)
    except KeyError as e:
        lprint(f"Faltan parámetros en la configuración ({conn_name}): {e}")
        return None

def engineCosteo(): #Costeo
    return openEngine('conn_costeo')


# #### Función que retorna el dataframe de una tabla
def fConsultaTabla(conexion, tabla):
    conx, ml = conexion()
    query = f"SELECT * FROM {ml}.{tabla}"
    df = psql.read_sql(query, conx)
    closeConnection(conx)
    lprint(f"DataFrame de tabla {ml}.{tabla}")
    return df


# #### Función que retorna los resultados de una consulta
def fConsultaScript(conexion, query):
    try:
        conx = conexion()[0]
        df = psql.read_sql(query, conx)
        return df
    except Exception as e:
        lprint(f"Error en la consulta SQL: {e}")
        return None
    finally:
        try:
            closeConnection(conx)
        except:
            pass


# #### Función que ejecuta una acción de la BD
def fEjecutaScript(conexion, query):
    conx = conexion()[0]
    cursor = conx.cursor()
    try:
      cursor.execute(query)
      conx.commit()
#      lprint("Ejecutar query")
    except Exception as e:
      conx.rollback()
      lprint(f"Error ejecutando query: {e}")
    finally:
      cursor.close()
      closeConnection(conx)


def fEjecutaDDL(conexion, ruta):
    lprint("Ejecucion DDL "+ruta)
    conx, esquema = conexion()
    with open(ruta, 'r', encoding='utf-8') as archivo:
        query = archivo.read()
        query = query.format(esquema=esquema)

    cursor = conx.cursor()
    try:
        cursor.execute(query)
        conx.commit()
        lprint("Instrucciones DDL ejecutadas con éxito.")
    except Exception as e:
        conx.rollback()
        print(f"Error ejecutando Instrucciones DDL: {e}")
    finally:
        cursor.close()
        closeConnection(conx)
    lprint("Instrucciones DDL finalizada")


# Función para insertar registro por registro
def fInsertaTabla(conexion, esquema, tabla, df):
    cursor = conexion.cursor()
    cols = ",".join([str(i) for i in df.columns.tolist()])
    for i, row in df.iterrows():
        sql = "INSERT INTO " + esquema + "." + tabla + " (" +cols+ ") VALUES (" + "%s,"*(len(row)-1) + "%s)"
        row = row.fillna(value=None)
        print(sql, tuple(row))
        cursor.execute(sql, tuple(row))
    conexion.commit()
    closeConnection(conexion)
    lprint(f"Datos insertados en tabla {esquema}.{tabla}")


# Función que ingresa los registros en la tabla de la BD
def fInsertaRegistros(conexion, esquema, tabla, df):
    cursor = conexion.cursor()
    cols = ",".join([str(i) for i in df.columns.tolist()])
    placeholders = ",".join(["%s"] * len(df.columns))
    consulta_sql = sql.SQL(f"INSERT INTO {esquema}.{tabla} ({cols}) VALUES ({placeholders})")
    df = df.astype(object).where(pd.notnull(df), None)
    datos = [tuple(x) for x in df.to_numpy()]
    cursor.executemany(consulta_sql, datos)
    conexion.commit()
    cursor.close()
    lprint(f"Datos insertados por registros en tabla {esquema}.{tabla}")


# #### Función que elimina todos los registros de una tabla de la BD
def fLimpiaTabla(conexion, esquema, tabla):
    cursor = conexion.cursor()
    cursor.execute(f"DELETE FROM {esquema}.{tabla}")
    conexion.commit()
    closeConnection(conexion)
    lprint(f"Eliminar registros de tabla {esquema}.{tabla}")


# #### Retorna consulta de una columna en una lista
def fRetornaLista(conexion,query):
    dfLista = fConsultaScript(conexion,query)
#    lprint(f"lista {str(dfLista.shape)}")
    ListaS = None
    if dfLista.shape[0] > 0:
        ListaS = ["'" + "','".join(dfLista[col].astype(str)) + "'" for col in dfLista.columns]
        if len(ListaS) == 1:
            ListaS = ListaS[0]
    return ListaS


# #### Crear tabla reemplazando
def fCrearTabla(miDF, miEngine, miEsquema, miTabla, miDtype=False):
    try:
        if miDtype:
          miDF.to_sql(miTabla, miEngine, schema=miEsquema, if_exists='replace', index=False, dtype=miDtype)
        else:
          miDF.to_sql(miTabla, miEngine, schema=miEsquema, if_exists='replace', index=False)
        res = 'Logrado'
    except Exception as e:
        res = e
    lprint('Tabla '+ str(miTabla) + ' creada' + str(res))
    return res


# #### Consulta desde un archivo .sql
def ConsultaSQL(conexion, ruta, convocatoria_id=None):
    lprint("Inicio consulta " + ruta)
    try:
        with open(ruta, 'r', encoding='utf-8') as archivo:
            query = archivo.read()
            query = query.format(esquema=conexion()[1], convocatoria_id=convocatoria_id)

        df = fConsultaScript(conexion, query)
        lprint("Fin consulta " + ruta + " " + str(df.shape))
        return df
    except Exception as e:
        lprint(f"Error general en ConsultaSQL: {e}")
        return None


# #### Retorna los datos de un Excel en un dataframe
def fLeerExcel(ruta, libro, hoja, fila1):
    lprint('EXCEL A LEER: '+ libro)
    return pd.read_excel(ruta + libro, sheet_name=hoja, header=fila1)


# #### Retirar tildes
def rRetirarTildes(value):
    return unidecode(str(value).strip().upper())


# Diccionario con los SMMMLV de los ultimos 12 años
def fSalarios():
    return fConsultaTabla(openCosteo,'salario_minimo')


# corregir el dato a entero
def fCorregirInt(columna,es_nulo=False):
    col = columna.astype(str).str.strip().apply(
      lambda x: int(float(re.sub(r'[^\d.-]', '', x))) if re.sub(r'[^\d.-]', '', x) != '' else (pd.NA if es_nulo else 0))
    if es_nulo:
        col = col.astype('Int64')
    else:
        col = col.astype('int32' if col.max() < pow(2, 30) else 'int64')
    return col

# busca el valor 2^n proximo
def rTecho(valor):
    return 2 ** math.ceil(math.log2(valor))
