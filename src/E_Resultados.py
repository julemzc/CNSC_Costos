#!/usr/bin/env python
# coding: utf-8

# Librerias generales
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
import zipfile
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib.backends.backend_pdf import PdfPages

# Funciones Generales
from src.A_Generales import lprint, fConsultaScript, openCosteo # type: ignore

# Exportar resultados de las proyecciones de los empleos en Excel
def fResultadosEmpleos(ejecucion):
    ml = openCosteo()[1]

    queryDB = f"""
    SELECT
    ns.escenario_id,
    TRANSLATE(substring(upper(ne.nombre),1,30),'/*+- _.,;()ÑÁÉÍÓÚ!"$%&=','') escenario,
    nr.id redneuronal_id,
    nr.nombre red,
    ne.divide_ascenso,
    ns.ranking::varchar
    FROM {ml}.nn_stats ns
    LEFT JOIN {ml}.np_escenas ne on (ns.escenario_id = ne.id)
    LEFT JOIN {ml}.np_redneuronal nr on (ns.redneuronal_id = nr.id)
    WHERE ejecucion_id = {str(ejecucion)}
    AND NOT ns.ascenso
    ORDER BY ns.escenario_id, ns.redneuronal_id
    """
  
    dfProy = fConsultaScript(openCosteo, queryDB)

    campos = ""
    condicion = ""
    for i, row in dfProy.iterrows():
        alias = str(row[1])+'_'+str(row[3])
        campos = campos + f"""
        {alias}.mun_inscritos {alias}_inscritos,
        {alias}.mun_aprobo_vrm {alias}_ins_vrm,
        {alias}.mun_aprobo_escritas {alias}_ins_esc,"""
        if bool(row[4]):
            condicion = condicion + f""" LEFT JOIN {ml}.nn_proyeccion {alias} on (ne.ejecucion_id = {alias}.ejecucion_id and ne.id = {alias}.nn_empleo_id AND {alias}.escenario_id = {str(row[0])} AND {alias}.redneuronal_id = {str(row[2])} AND {alias}.ascenso = ne.concurso_ascenso ) """
        else:
            condicion = condicion + f""" LEFT JOIN {ml}.nn_proyeccion {alias} on (ne.ejecucion_id = {alias}.ejecucion_id and ne.id = {alias}.nn_empleo_id AND {alias}.escenario_id = {str(row[0])} AND {alias}.redneuronal_id = {str(row[2])} ) """

    queryEx = f"""
    SELECT
    ne.empleo_id,
    ne.concurso_ascenso,
    ne.asignacion_salarial,
    ne.agno,
    ne.smmlv,
    ne.nivel,
    ne.grado,
    ne.denominacion,
    ne.conv_padre,
    ne.conv_nombre,
    ne.entidad,
    ne.tipo_entidad,
    ne.departamento,
    ne.municipio,
    ne.codigo_dane,
    ne.mun_categoria,
    ne.vacantes_opec,
    ne.vacantes,
    ne.reqs_estudio,
    ne.experiencia,
    ne.sin_experiencia,
    {campos}
    ne.empleo_id
    FROM {ml}.nn_empleo ne
    {condicion}
    WHERE ne.ejecucion_id = {str(ejecucion)}"""
    dfExcel = fConsultaScript(openCosteo, queryEx)
    lprint(queryEx)
    lprint("Tabla de resultados: "+str(dfExcel.shape))
    return dfExcel


# Exportar los diccionarios de la ejecucion
def fResultadosEjecucion(ejecucion):
    ml = openCosteo()[1]
    query = f"""
    SELECT * 
    FROM {ml}.np_ejecucion
    WHERE id = {str(ejecucion)}"""
    dfDicc = fConsultaScript(openCosteo, query)
    return dfDicc


# Exportar los resultados de la ejecución en el Excel
def fResultadosExcel(ejecucion, nombre):
    dfExcel = fResultadosEmpleos(ejecucion)
    dfDicc = fResultadosEjecucion(ejecucion)
    with pd.ExcelWriter(nombre+".xlsx", engine="xlsxwriter") as writer:
        dfExcel.to_excel(writer, sheet_name="BASE", index=False)
        dfDicc.to_excel(writer, sheet_name="Dicc", index=False)
  #  lprint("Excel con los datos creado "+os.get+nombre+".xlsx")


# Grafica de epocas del MAE y Valores de perdida
def fGraficaEpocas(Dict, pdf_pages, opc):
    color1 = ['r','g','b','y','c','m']
    labelRed = []
    label = []
    escena = list(Dict.keys())[0]
    for i, r in enumerate(Dict[escena]['Historico']):
        labelRed.append(str(Dict[escena]['Historico'][i]['red'][6]))
        label.append(labelRed[i] + " "+opc)
        label.append(labelRed[i] + " v"+opc)

    for e in Dict:
        fig = plt.figure(figsize=(6,3))
        l0 = 0
        l1 = 300
        for i, hist in enumerate(Dict[e]['Historico']):
            plt.plot(range(1,len(hist[opc][l0:l1])+1), hist[opc][l0:l1], '-',label=opc, color=color1[i])
            plt.plot(range(1,len(hist['v'+opc][l0:l1])+1), hist['v'+opc][l0:l1], '--',label='val '+opc, color=color1[i])
        if opc == 'mae':
            plt.title('Error absoluto medio - '+ Dict[e]['Filtro'])
        if opc == 'loss':
            plt.title('Valor Perdida - '+ Dict[e]['Filtro'])
        plt.ylabel(opc)
        plt.xlabel('Epocas')
        plt.grid(True)
        plt.ylim(bottom=0)
        plt.legend(label, loc='best', fontsize = 'x-small')
        pdf_pages.savefig(fig, bbox_inches='tight')


# Graficas de Datos de muestra vs Datos de proyección
def fGraficasRelacion(Dict, pdf_pages):
    for e in Dict:
        fig = plt.figure(figsize=(3,3))
        a = plt.axes(aspect='equal')
        Dict[e]['test_targets']
        rango = Dict[e]['test_targets'].shape[0]
        plt.scatter(Dict[e]['test_targets'],Dict[e]['train_targets'][0:rango])
        plt.title('Relación - '+ Dict[e]['Filtro'])
        plt.xlabel('Real')
        plt.ylabel('Prediccion')
        plt.grid(True)
        lims = [0,(Dict[e]['dfEscenario']['mun_inscritos'].max())*0.75]
        plt.xlim(lims)
        plt.ylim(lims)
        plt.plot(lims,lims)
        pdf_pages.savefig(fig, bbox_inches='tight')

        
# Correlación de las variables seleccionadas en los escenarios
def fGraficasCorrEscenas(Dict, pdf_pages):
    for e in Dict:
        dfCorr = pd.concat([Dict[e]['DatosX'], pd.DataFrame(Dict[e]['DatosY'], columns=['inscritos'])], axis=1)
        etiquetas = dfCorr.columns.tolist()
#        dfCorr = df.select_dtypes(include=['float64', 'int32', 'Int64'])
        fig = plt.figure(figsize=(dfCorr.shape[1],dfCorr.shape[1]))
        sns.set(font_scale=1)
        hm = sns.heatmap(dfCorr.corr().to_numpy(),
                         cbar=False,
                        annot=True,
                        square=True,
                        fmt='.2f',
                        annot_kws={'size': 10},
                        yticklabels=etiquetas,
                        xticklabels=etiquetas)
        hm.set_title(Dict[e]['Filtro'], fontsize =16)
        pdf_pages.savefig(fig, bbox_inches='tight')


# Correlación de la base completa de SIMO
def fGraficasCorrSimo(dfEmpleo, pdf_pages):
    dfCorr = dfEmpleo.select_dtypes(include=['float64', 'int32', 'Int64'])
    fig = plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    etiquetas = dfCorr.corr().columns.tolist()
    hm = sns.heatmap(dfCorr.corr().to_numpy(),
                    cbar=False,
                    annot=True,
                    square=True,
                    fmt='.2f',
                    annot_kws={'size': 10},
                    yticklabels=etiquetas,
                    xticklabels=etiquetas)
    hm.set_title('SIMO', fontsize =12)
    pdf_pages.savefig(fig, bbox_inches='tight')


# Graficas para el PDF
def fResultadosGraficas(Dict, dfEmpleo, nombre):
    # Crear un objeto PdfPages para guardar las gráficas en un archivo PDF
    pdf_pages = PdfPages(nombre+".pdf")
    fGraficaEpocas(Dict,pdf_pages,'mae')
    fGraficaEpocas(Dict,pdf_pages,'loss')
#    fGraficasRelacion(Dict, pdf_pages)
    fGraficasCorrEscenas(Dict, pdf_pages)
#    fGraficasCorrSimo(dfEmpleo, pdf_pages)
    pdf_pages.close()
    lprint("PDF con las graficas creado "+nombre+".pdf")
    

# Valores OLS - Mínimos cuadrados ordinarios
def fAnalisisOLS(Dict, listaArchivo):
    for e in Dict:
        Dict[e]['DatosX'].columns = Dict[e]['DatosX'].columns.str.replace(' ', '_')
        col_names = Dict[e]['DatosX'].columns.tolist()
        modelo = 'inscritos ~ ' + ' + '.join(col_names)
        listaArchivo.append(modelo)
    
        df = pd.concat([pd.DataFrame(Dict[e]['DatosY'],columns=['inscritos']), Dict[e]['DatosX']], axis=1)
        listaArchivo.append(str(Dict[e]['Filtro']))
        try:
            lm = sm.ols(formula = modelo, data = df).fit()
            listaArchivo.append(str(lm.summary()))
            lprint(str(lm.summary()))
        except ValueError:
            listaArchivo.append('ERROR')
            pass


# Valores de los Percentiles de las variables y los inscritos
def fAnalisisPercentil(Dict, listaArchivo):
    for e in Dict:
        listaArchivo.append(Dict[e]['Filtro'])
        listaArchivo.append(Dict[e]['dfModelo'].describe())
        listaArchivo.append(Dict[e]['dfEscenario']['mun_inscritos'].describe().T)


# Valores de los valores R2, MSE, MAE
def fAnalisisR2(Dict, listaArchivo):
    for e in Dict:
        for i, hist in enumerate(Dict[e]['Historico']):
            Xmodel = hist['model'].predict([Dict[e]['X']])
            Yseries = pd.Series(Dict[e]['Y'].flatten())
            listaArchivo.append(Dict[e]['Filtro'])
            listaArchivo.append("Red Neuronal "+ hist['red'][6])
            listaArchivo.append(f"Coeficiente de Determinación por r2_score: {r2_score(Yseries, Xmodel)*100:0.3f}%")
            listaArchivo.append(f"Test Mean Squared Error (MSE): {mean_squared_error(Yseries, Xmodel):0.3f}")
            listaArchivo.append(f"Test Mean Absolute Error (MAE): {mean_absolute_error(Yseries, Xmodel):0.3f}")


# Exportar en txt
def fResultadosTexto(Dict, nombre):
    listaArchivo = []
    fAnalisisOLS(Dict, listaArchivo)
    fAnalisisPercentil(Dict, listaArchivo)
    fAnalisisR2(Dict, listaArchivo)
    with open(nombre+".txt", "w")as archivo:
        for texto in listaArchivo:
            archivo.write(str(texto) + '\n\n')
    lprint("Archivo exportado: "+nombre+".txt")


# Crea un archivo ZIP en modo escritura
def rArchivoZip(nombre):
    # Nombre del archivo ZIP de salida
    nombreZip = nombre +".zip"
    archivos = [nombre+".xlsx", nombre+".pdf", nombre+".txt"]

    with zipfile.ZipFile(nombreZip, "w") as archivo_zip:
        for archivo in archivos:
            archivo_zip.write(archivo)
    lprint(f"Se han adjuntado {len(archivos)} archivos al archivo ZIP: {nombreZip}")
