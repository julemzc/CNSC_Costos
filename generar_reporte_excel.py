#!/usr/bin/env python
# coding: utf-8

"""
Script: generar_reporte_excel.py
Descripción: Genera reporte Excel de una ejecución específica
Uso: python generar_reporte_excel.py <ejecucion_id>
Ejemplo: python generar_reporte_excel.py 145
"""

import sys
import os
import time
from pathlib import Path

# Importar funciones necesarias
from src.A_Generales import load_config, lprint, fConsultaScript, openCosteo
from src.E_Resultados import fResultadosExcel

# Valida que se haya proporcionado el ID de ejecución
def validar_argumentos():
    if len(sys.argv) < 2:
        lprint("ERROR: Falta el ID de ejecución")
        lprint("Uso correcto:")
        lprint("Python generar_reporte_excel.py <ejecucion_id>")
        sys.exit(1)
    
    try:
        ejecucion_id = int(sys.argv[1])
        return ejecucion_id
    except ValueError:
        lprint(f"ERROR: '{sys.argv[1]}' no es un número válido")
        lprint("El ID de ejecución debe ser un número entero")
        sys.exit(1)

# Verifica que la ejecución exista en la base de datos
def verificar_ejecucion_existe(ejecucion_id):
    ml = openCosteo()[1]
    query = f"""
    SELECT id, fecha_creacion, filtro 
    FROM {ml}.np_ejecucion 
    WHERE id = {ejecucion_id}
    """
    resultado = fConsultaScript(openCosteo, query)
    
    if resultado is None or resultado.empty:
        lprint(f"ERROR: La ejecución {ejecucion_id} no existe en la base de datos")
        lprint("\nVerifica que:")
        lprint("  1. El ID sea correcto")
        lprint("  2. La ejecución se haya completado exitosamente")
        lprint("  3. Tengas acceso a la base de datos")
        sys.exit(1)
    
    return resultado.iloc[0]

# Cuenta el número de empleos asociados a la ejecución
def contar_empleos(ejecucion_id):
    ml = openCosteo()[1]
    query = f"""
    SELECT COUNT(*) as total 
    FROM {ml}.nn_empleo 
    WHERE ejecucion_id = {ejecucion_id}
    """
    resultado = fConsultaScript(openCosteo, query)
    
    if resultado is not None and not resultado.empty:
        return resultado.iloc[0]['total']
    return 0

# Genera el archivo Excel con los resultados de la ejecución
def generar_excel(ejecucion_id):
    # Crear nombre base para el archivo
    config = load_config()
    output_dir = config.get('output_dir', 'output/')
    
    # Crear directorio si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    nombre = output_dir + f'Resultados_{ejecucion_id}_{time.strftime("%y%m%d_%H%M%S")}'
    
    lprint(f"Generando Excel...")
    lprint(f" Ubicación: {nombre}.xlsx")

    try:
        fResultadosExcel(ejecucion_id, nombre)
        # Verificar que se haya creado el archivo
        archivo_excel = f"{nombre}.xlsx"
        if os.path.exists(archivo_excel):
            tamano = os.path.getsize(archivo_excel) / 1024  # KB
            lprint(f"\n Excel generado exitosamente")
            lprint(f" Archivo: {archivo_excel}")
            lprint(f" Tamaño: {tamano:.1f} KB")
            return archivo_excel
        else:
            lprint(f"\n El archivo no se generó correctamente")
            return None

    except Exception as e:
        lprint(f"\n Error generando Excel: {e}")
        lprint(f"Error generando Excel para ejecución {ejecucion_id}: {e}")
        return None

# Función principal
def main():
    lprint("  GENERADOR DE REPORTE EXCEL - Sistema de Proyección")
    
    # 1. Validar argumentos
    ejecucion_id = validar_argumentos()
    lprint(f"Ejecución ingresada: {ejecucion_id}")
    
    # 2. Verificar que la ejecución existe
    lprint(f"Verificando existencia de la ejecución...")
    info_ejecucion = verificar_ejecucion_existe(ejecucion_id)
    lprint(f"Ejecución encontrada")
    lprint(f" Fecha: {info_ejecucion['fecha_creacion']}")
    lprint(f" Filtro: {info_ejecucion['filtro']}")

    # 3. Contar empleos
    lprint(f"Consultando datos...")
    total_empleos = contar_empleos(ejecucion_id)
    lprint(f" Total de empleos: {total_empleos}")
    if total_empleos == 0:
        lprint(f"\nADVERTENCIA: La ejecución no tiene empleos registrados")
        lprint(f"El Excel podría estar vacío")
    
    # 4. Generar Excel
    archivo = generar_excel(ejecucion_id)
    
    # 5. Resumen final
    if archivo:
        lprint("GENERACIÓN COMPLETADA EXITOSAMENTE")
        lprint(f"\n Archivo generado: {archivo} /// contiene {total_empleos} empleos proyectados")
    else:
        lprint("GENERACIÓN FALLIDA")
        lprint(f"\n No se pudo generar el archivo Excel / Revisa los logs para más detalles")
        sys.exit(1)
    
    lprint("\n" + "=" * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        lprint("\n\n  Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        lprint(f"\n ERROR INESPERADO: {e}")
        lprint(f"Error inesperado en generar_reporte_excel.py: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)