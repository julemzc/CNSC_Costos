#!/usr/bin/env python
# coding: utf-8

# # Red neuronal para proyectar los inscritos para una convocatoria

#librerias
import os
import pandas as pd
import time
from datetime import datetime

# Maximo de columnas
pd.options.display.max_columns = None
import warnings
warnings.filterwarnings('ignore')

# Importar funciones
from src.A_Generales import lprint, load_config
from src.B_Historico import rDatosSimo, rConvocatoriaSimo
from src.C_RedNeuronal import fEscenarios, fEntrenamiento, fProyeccion
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
        dfCostos, ejecucion = fGuardarResultados(Dict, dfCostos, Convocatoria, co)
        lprint("FIN - Resultados guardados en la BD \n")

        lprint("Inicio - Exportar resultados a Excel y PDF")
        nombre = load_config()['output_dir'] + 'Resultados_' + str(ejecucion)+"_"+time.strftime("%y%m%d_%H%M%S")
        fResultadosExcel(ejecucion, nombre)
        fResultadosGraficas(Dict, dfEmpleo, nombre)
        fResultadosTexto(Dict, nombre)
        rArchivoZip(nombre)
        lprint(f"FIN - Resultados en Excel y PDF {nombre}")
    else:
        lprint("Convocatoria NO seleccionada")

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
