#!/usr/bin/env python
# coding: utf-8

# # Red neuronal para proyectar los inscritos para una convocatoria

#librerias
import os
import pandas as pd
import time
from datetime import datetime
import streamlit as st


# Maximo de columnas
pd.options.display.max_columns = None
import warnings
warnings.filterwarnings('ignore')

# Importar funciones
from src.A_Generales import load_config, fConsultaScript, openCosteo
from src.B_Historico import rDatosSimo, rConvocatoriaSimo
from src.C_RedNeuronal import fEntrenamiento, fProyeccion
from src.D_GuardarBD import fGuardarResultados
from src.E_Resultados import fResultadosExcel, fResultadosTexto, fResultadosGraficas, rArchivoZip


def pipeline():
    st.title("Ejecución de Red Neuronal")

    query = "SELECT convocatoria_id, nombre FROM seleccion_convocatoria WHERE NOT proyectar ORDER BY nombre"
    dfSC = fConsultaScript(openCosteo, query)
    listaSC = dict(zip(dfSC['convocatoria_id'], dfSC['nombre']))
    id_convocatoria = st.selectbox(
        "Selecciona Convocatoria:",
        options=list(listaSC.keys()),
        format_func=lambda x: listaSC[x]  # Muestra el nombre pero retorna el ID
    )
    st.write(f"ID seleccionado: {id_convocatoria}")
    
    st.success(f"Inicio del Pipeline")
    if st.button("Ejecutar"):
        progress_area = st.empty()
        progress_area.text("Inicio - Lectura Base de Datos")
        dfEmpleo = rDatosSimo()
        progress_area.text("FIN - Lectura Base de Datos")

        progress_area.text("Inicio - Realizar entrenamiento de escenarios")
        Dict = fEntrenamiento(dfEmpleo, False)
        progress_area.text("FIN - Realizar entrenamiento de escenarios")

        progress_area.text("Inicio - Seleccionar convocatoria a proyectar")
        dfCostos, co = rConvocatoriaSimo(id_convocatoria)
        progress_area.text("FIN - Seleccionar convocatoria a proyectar")

        if not dfCostos.empty:
            progress_area.text(f"Inicio - Realizar proyección de convocatoria {id_convocatoria}")
            dfCostos, Convocatoria = fProyeccion(Dict, dfCostos)
            progress_area.text("FIN - Proyección de inscritos finalizada \n")
            
            progress_area.text("Inicio - Guardar resultados en la BD")
            dfCostos, ejecucion = fGuardarResultados(Dict, dfCostos, Convocatoria, co)
            progress_area.text("FIN - Resultados guardados en la BD \n")

            progress_area.text("Inicio - Exportar resultados a Excel y PDF")
            nombre = load_config()['output_dir'] + 'Resultados_' + str(ejecucion)+"_"+time.strftime("%y%m%d_%H%M%S")
            fResultadosExcel(ejecucion, nombre)
            fResultadosGraficas(Dict, dfEmpleo, nombre)
            fResultadosTexto(Dict, nombre)
            rArchivoZip(nombre)
            progress_area.text(f"FIN - Resultados en Excel y PDF {nombre}")
        else:
            progress_area.text("Convocatoria NO seleccionada")

    st.success(f"FIN del Pipeline")

## Ejecución de la red neuronal
if __name__ == '__main__':
    pipeline()