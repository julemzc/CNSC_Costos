#!/usr/bin/env python
# coding: utf-8

# # Red neuronal para proyectar los inscritos para una convocatoria
# Autor: Julián Leonardo Martínez C.
# Fecha: Diciembre de 2025
# Descripción: Este script implementa una red neuronal para proyectar los inscritos en una convocatoria específica utilizando datos históricos de empleo. El proceso incluye la lectura de datos, entrenamiento del modelo, proyección de resultados y almacenamiento de los mismos en una base de datos, así como la generación de reportes en Excel y PDF.

#librerias
import pandas as pd
import warnings
import os
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

# Importar funciones
from src.A_Generales import *
from src.B_Historico import *
from src.C_RedNeuronal import fEntrenamiento, fProyeccion
from src.D_GuardarBD import fGuardarResultados
from src.E_Resultados import fResultadosExcel, fResultadosTexto, fResultadosGraficas, rArchivoZip


def pipeline(id_convocatoria):
    lprint("Inicio - Lectura Base de Datos \n")
    dfEmpleo = rDatosSimo()
    lprint("FIN - Lectura Base de Datos")

    lprint("Inicio - Realizar entrenamiento de escenarios")
    Dict = fEntrenamiento(dfEmpleo, False)
    lprint("FIN - Realizar entrenamiento de escenarios")

    lprint("Inicio - Seleccionar convocatoria a proyectar")
    dfCostos, co = rConvocatoriaSimo(id_convocatoria)
    lprint("FIN - Seleccionar convocatoria a proyectar")

    if not dfCostos.empty:
        lprint(f"Inicio - Realizar proyección de convocatoria {id_convocatoria}")
        dfCostos, Convocatoria = fProyeccion(Dict, dfCostos)
        lprint("FIN - Proyección de inscritos finalizada \n")
        
        lprint("Inicio - Guardar resultados en la BD")
        dfCostos, ejecucion, ruta = fGuardarResultados(Dict, dfCostos, Convocatoria, co)
        lprint("FIN - Resultados guardados en la BD \n")

        lprint("Inicio - Exportar resultados a Excel y PDF")
        fResultadosExcel(ejecucion, ruta)
        fResultadosTexto(Dict, ruta)
        fResultadosGraficas(Dict, dfEmpleo, ruta)
        rArchivoZip(ruta)
        lprint(f"FIN - Resultados en Excel y PDF {ruta}")
    else:
        lprint(f"Convocatoria {id_convocatoria} NO seleccionada")

    lprint("FIN del Pipeline")

## Ejecución de la red neuronal
if __name__ == '__main__':
    lprint("Directorio actual:", os.getcwd())
    try:
        lprint("Inicia Pipeline \n\n")
        id_convocatoria # type: ignore
    except NameError:
        id_convocatoria = 0
    lprint(f'La convocatoria seleccionada es: {id_convocatoria}')
    pipeline(id_convocatoria)
