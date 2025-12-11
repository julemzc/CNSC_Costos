#!/usr/bin/env python
# coding: utf-8

# Librerias generales
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
import zipfile
import locale
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib.backends.backend_pdf import PdfPages

locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')

# Funciones Generales
from src.A_Generales import lprint, fConsultaScript, openCosteo # type: ignore

# Exportar resultados de las proyecciones de los empleos en Excel
def fResultadosEmpleos(ejecucion):
    lprint("RESULTADOS DE EMPLEO")
    ml = openCosteo()[1]

    queryDB = f"""
    SELECT
    ns.escenario_id,
    TRANSLATE(substring(upper(ne.nombre),1,30),'/*+- _.,;()ÑÁÉÍÓÚ!"$%&=','') escenario,
    nr.id redneuronal_id,
    nr.nombre red,
    ne.divide_ascenso,
    regexp_replace(initcap(ne.nombre), '[\n\r\t,;|]+', ' ', 'g') esc_nombre,
    ns.ranking::varchar
    FROM {ml}.nn_stats ns
    LEFT JOIN {ml}.np_escenas ne on (ns.escenario_id = ne.id)
    LEFT JOIN {ml}.np_redneuronal nr on (ns.redneuronal_id = nr.id)
    WHERE ejecucion_id = {str(ejecucion)}
    AND NOT ns.ascenso
    ORDER BY ns.escenario_id, ns.redneuronal_id
    """

    queryDB = f"""
    SELECT DISTINCT
    trim(esc ->> 'escena')::int4 AS esc_id,
    TRANSLATE(substring(upper(esc ->> 'nombre'),1,30),'/*+- _.,;()ÑÁÉÍÓÚ!"$%&=','') escenario,
    trim(red ->> 'id')::int4 AS red_id,
    trim(red ->> 'nombre') AS red,
    trim(esc ->> 'divide_ascenso')::bool AS div_ascenso,
    regexp_replace(initcap(trim(esc ->> 'nombre')), '[\n\r\t,;|]+', ' ', 'g') esc_nombre
    FROM public.nn_stats ns,
    public.np_ejecucion ne,
    json_array_elements(ne.escenarios::json) as esc,
    json_array_elements(ne.redneuronal::json) as red
    WHERE ns.ejecucion_id = ne.id AND ns.ejecucion_id = {str(ejecucion)}
    AND NOT ns.ascenso
    ORDER BY 1,3;
    """
    dfProy = fConsultaScript(openCosteo, queryDB)
    campos = ""
    condicion = ""
    for row in dfProy.itertuples():
        alias = f"{str(row.escenario)}_{str(row.red)}"
        esc_nombre = f"{str(row.esc_nombre)} ({str(row.red)})"
        lprint(esc_nombre)
        campos = campos + f"""
        {alias}.mun_inscritos "Inscritos / {esc_nombre}",
        {alias}.mun_aprobo_vrm "Aprueba VRM / {esc_nombre}",
        {alias}.mun_aprobo_escritas "Aprueba Escritas / {esc_nombre}",
        {alias}.mun_pcd_inscritos "Inscritos PCD / {esc_nombre}","""
        if bool(row.div_ascenso):
            condicion = condicion + f""" LEFT JOIN {ml}.nn_proyeccion {alias} on (ne.ejecucion_id = {alias}.ejecucion_id and ne.id = {alias}.nn_empleo_id AND {alias}.escenario_id = {row.esc_id} AND {alias}.redneuronal_id = {row.red_id} AND {alias}.ascenso = ne.concurso_ascenso ) """
        else:
            condicion = condicion + f""" LEFT JOIN {ml}.nn_proyeccion {alias} on (ne.ejecucion_id = {alias}.ejecucion_id and ne.id = {alias}.nn_empleo_id AND {alias}.escenario_id = {row.esc_id} AND {alias}.redneuronal_id = {row.red_id} ) """

    queryEx = f"""
    SELECT
    ne.empleo_id "OPEC",
    CASE WHEN ne.concurso_ascenso THEN 'Ascenso' ELSE 'Ingreso' END "Ascenso / Ingreso",
    ne.asignacion_salarial "Salario",
    ne.agno "Año",
    ne.smmlv "SMMLV",
    ne.nivel "Nivel",
    ne.grado "Grado",
    ne.denominacion "Denominación",
    ne.conv_padre "Proceso de Selección",
    ne.conv_nombre "Convocatoria",
    ne.entidad "Entidad",
    ne.tipo_entidad "Tipo de Entidad",
    ne.departamento "Departamento",
    ne.municipio "Municipio",
    ne.codigo_dane "Código DANE",
    ne.mun_categoria "Categoría Municipio",
    ne.vacantes_opec "Total Vacantes OPEC",
    ne.vacantes "Vacantes Municipio",
    ne.reqs_estudio "Requisitos de Estudio",
    CASE WHEN nbc_tecnico IS NOT NULL THEN 'TÉCNICO: '||nbc_tecnico||' /// 'ELSE '' END ||
    CASE WHEN nbc_esp_tecnico IS NOT NULL THEN 'TÉCNICO ESP.: '||nbc_esp_tecnico||' /// ' ELSE '' END ||
    CASE WHEN nbc_tecnologico IS NOT NULL THEN 'TECNOLOGICO: '||nbc_tecnologico||' /// ' ELSE '' END ||
    CASE WHEN nbc_esp_tecnologico IS NOT NULL THEN 'TECNOLOGICO ESP.: '||nbc_esp_tecnologico||' /// ' ELSE '' END ||
    CASE WHEN nbc_profesional IS NOT NULL THEN 'PROFESIONAL: '||nbc_profesional||' /// ' ELSE '' END ||
    CASE WHEN nbc_esp_profesional IS NOT NULL THEN 'PROFESIONAL ESP.: '||nbc_esp_profesional||' /// ' ELSE '' END ||
    CASE WHEN nbc_maestria IS NOT NULL THEN 'MAESTRIA: '||nbc_maestria||' /// ' ELSE '' END ||
    CASE WHEN nbc_doctorado IS NOT NULL THEN 'DOCTORADO: '||nbc_doctorado||' /// ' ELSE '' END AS "Nucleo Básico de Conocimiento",
    ne.experiencia "Experiencia",
    CASE WHEN ne.sin_experiencia IS NULL THEN '' WHEN ne.sin_experiencia THEN 'SI' ELSE 'NO' END AS "Sin Experiencia",
    {campos}
    ne.empleo_id "OPEC"
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

def fVistaInscritosNivel(ejecucion):
    ml = openCosteo()[1]
    query = f"""
    SELECT
    trim(escenario) "Escenario",
    red "Red Neuronal",
    CASE WHEN elegida THEN 'Sí' ELSE 'No' END AS "Modelo Elegido",
    vacantes "Total Vacantes",
    inscritos "Total Inscritos",
    aprobo_vrm "Total Aprueban VRM",
    aprobo_escritas "Total Aprueban Escritas",
    pcd_inscritos "Total Inscritos PCD",
    vac_ascenso "Vacantes Ascenso",
    ins_ascenso "Inscritos Ascenso",
    vac_ingreso "Vacantes Ingreso",
    ins_ingreso "Inscritos Ingreso",
    vac_asesor "Vacantes Asesor",
    ins_asesor "Inscritos Asesor",
    vac_profesional "Vacantes Profesional",
    ins_profesional "Inscritos Profesional",
    vac_tecnico "Vacantes Técnico",
    ins_tecnico "Inscritos Técnico",
    vac_asistencial "Vacantes Asistencial",
    ins_asistencial "Inscritos Asistencial",
    vac_otros_nivel1 "Vacantes Otros Nivel 1",
    ins_otros_nivel1 "Inscritos Otros Nivel 1",
    vac_otros_nivel2 "Vacantes Otros Nivel 2",
    ins_otros_nivel2 "Inscritos Otros Nivel 2",
    fecha_creacion "Fecha de Creación"
    FROM {ml}.vw_inscritos_nivel 
    WHERE ejecucion_id = {str(ejecucion)}"""
    df = fConsultaScript(openCosteo, query)
    return df


# Exportar los resultados de la ejecución en el Excel
def fResultadosExcel(ejecucion, nombre):
    lprint("entra a resultados excel")
    dfExcel = fResultadosEmpleos(ejecucion)
    lprint("dfExcel creado")
    dfDicc = fResultadosEjecucion(ejecucion)
    dfVista = fVistaInscritosNivel(ejecucion)
    with pd.ExcelWriter(nombre+".xlsx", engine="xlsxwriter") as writer:
        # Primera hoja con los resultados de la proyección de empleos
        dfExcel.to_excel(writer, sheet_name="BASE", index=False, startrow=2)

        workbook = writer.book
        worksheet = writer.sheets['BASE']
        title_format = workbook.add_format({ 'bold': True, 'font_size': 14, 'align': 'center', 'fg_color': '#203864', 'font_color': 'white'})
        number_format = workbook.add_format({ 'num_format': '#,##0' })
        money_format = workbook.add_format({ 'num_format': '$ #,##0' })
        row_json = workbook.add_format({ 'text_wrap': True, 'valign': 'top', 'align': 'left' })
        header_format = workbook.add_format({ 'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center', 'fg_color': '#4472C4', 'font_color': 'white', 'border': 1 })
        vacante_header = workbook.add_format({ 'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center', 'fg_color': "#FABF40", 'font_color': 'black', 'border': 1 })
        inscrito_header = workbook.add_format({ 'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center', 'fg_color': "#61CF2F", 'font_color': 'black', 'border': 1})

        worksheet.merge_range('A1:J1', f'PROYECCIÓN DE INSCRITOS - Ejecución #{ejecucion}   |   |   |   {datetime.now().strftime("%d de %B de %Y %H:%M")}', title_format)
        for col_num, value in enumerate(dfExcel.columns.values):
            worksheet.write(2, col_num, value, header_format)
        worksheet.set_row(2, 60)

        # Ajustar columnas
        worksheet.set_column('A:B', 8)
        worksheet.set_column('C:C', 12, money_format)
        worksheet.set_column('D:E', 7)
        worksheet.set_column('F:F', 12)
        worksheet.set_column('G:G', 6)
        worksheet.set_column('H:K', 15)
        worksheet.set_column('L:N', 13)
        worksheet.set_column('O:R', 9)
        worksheet.set_column('S:U', 25)
        worksheet.set_column('V:V', 6)
        worksheet.set_column('W:AZ', 12, number_format)

        worksheet.freeze_panes(3, 0)
        worksheet.autofilter(2, 0, len(dfExcel) + 3, len(dfExcel.columns) - 1)

        # Segunda hoja con el diccionario de la ejecución
        dfDicc.to_excel(writer, sheet_name="Dicc", index=False)
        worksheetD = writer.sheets['Dicc']
        worksheetD.set_column('B:B', 20)
        worksheetD.set_column('E:G', 35)
        worksheetD.set_row(1, 200, row_json)
        
        # Tercera hoja con el resumen de inscritos
#        cols_inscritos = [col for col in dfExcel.columns if 'Inscritos' in col or 'Aprueba' in col]
#        resumen = {col: dfExcel[col].sum() for col in cols_inscritos}
#        df_resumen = pd.DataFrame(list(resumen.items()), columns=['Inscritos / Convocatoria', 'Total'])
#        df_resumen.to_excel(writer, sheet_name='Resumen', index=False)
#        worksheetR = writer.sheets['Resumen']
#        worksheetR.set_column('A:A', 50)

        # Cuarta hoja con la vista de inscritos por nivel
        dfVista.to_excel(writer, sheet_name="InscritosNivel", index=False)
        worksheetV = writer.sheets['InscritosNivel']
        worksheetV.set_column('A:A', 30)
        worksheetV.set_column('Y:Y', 18)
        worksheetV.set_row(0, 45, row_json)
        worksheetV.freeze_panes(1, 0)
        for col_num, value in enumerate(dfVista.columns.values):
            if 'vacante' in value.lower():
                worksheetV.write(0, col_num, value, vacante_header)
            elif 'inscrito' in value.lower():
                worksheetV.write(0, col_num, value, inscrito_header)
            else:
                worksheetV.write(0, col_num, value, header_format)


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
        """
        Dict[e]['DatosX'].columns = Dict[e]['DatosX'].columns.str.replace(' ', '_')
        col_names = Dict[e]['DatosX'].columns.tolist()
        """
        col_names = Dict[e]['dfModelo'].columns.tolist()
        col_names.remove('mun_inscritos')

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
