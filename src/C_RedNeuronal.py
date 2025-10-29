#!/usr/bin/env python
# coding: utf-8

# #### Librerias generales
import pandas as pd
import numpy as np
import pickle
import time
import ast
import os
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature

# Librerias redes neuronales
from sklearn.preprocessing import StandardScaler #, PowerTransformer, QuantileTransformer, RobustScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold

from keras.layers import Dense, Dropout, BatchNormalization # type: ignore #, Normalization, 
from keras.models import Sequential # type: ignore
from keras.optimizers import Adam # type: ignore #, RMSprop
from keras.regularizers import l1_l2 # type: ignore #,l1, l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore

# Funciones Generales
from src.A_Generales import lprint, fConsultaScript, openCosteo, rRetirarTildes, rTecho # type: ignore
from src.B_Historico import fVariables, fAgregarSmmlvAgno # type: ignore

# carpeta de mlflow
def configurar_mlflow(nombre_carpeta="mlruns"):
    base_path = os.getcwd() # Carpeta base: el cwd del notebook/script
    mlruns_path = os.path.join(base_path, nombre_carpeta) # Ruta completa donde se guardará mlruns
    os.makedirs(mlruns_path, exist_ok=True) # Crear carpeta si no existe
    mlflow.set_tracking_uri(f"file:///{mlruns_path}")
    lprint(f"MLflow configurado en: {mlruns_path}")
    mlflow.set_experiment("Costos_"+time.strftime("%Y%m%d"))
    return mlruns_path


# leer los escenarios
def fEscenarios(dfEmpleo):
    lprint("Inicio - Creación de escenarios en base al historico de Simo")
    Dict = {}
    EscenaId = '-'
    ml = openCosteo()[1]
    query = f"""
    SELECT
    es.id, es.nombre, de.campo, de.operador,
    de.valor, de.percmin, de.percmax, COALESCE(de.concatena,''),
    COUNT(*) FILTER (WHERE de.activo) OVER (PARTITION BY de.escena_id) AS cont, de.id, es.divide_ascenso
    FROM {ml}.np_escenas es
    JOIN {ml}.np_escenas_detalle de on (es.id = de.escena_id)
    WHERE es.activo AND de.activo
    ORDER BY de.escena_id, de.concatena
    """
    dfEscenariosBD = fConsultaScript(openCosteo, query)

    for i, row in dfEscenariosBD.iterrows():
        lprint(row[1])
        if row[0] != EscenaId:
            cont = 1
            dfEscena = None
            dfEscena = fFiltroEscenario(dfEmpleo, row[2], row[3], row[4])
            lprint(f"Primer Fila {str(dfEscena.shape)} / {row[0]} / {row[1]} / {row[2]} / {row[3]} / {row[4]} / {row[5]} / {row[6]} / {row[7]}")
        else:
            cont += 1
            if row[7] == 'AND':
                dfEscena = fFiltroEscenario(dfEscena, row[2], row[3], row[4])
                lprint(f"AND {str(dfEscena.shape)} / {row[0]} / {row[1]} / {row[2]} / {row[3]} / {row[4]} / {row[5]} / {row[6]} / {row[7]}")
            if row[7] == 'OR':
                dfEscenaOR = fFiltroEscenario(dfEmpleo, row[2], row[3], row[4])
                dfEscena = pd.concat([dfEscena, dfEscenaOR], ignore_index=True)
                lprint(f"OR {str(dfEscena.shape)} / {row[0]} / {row[1]} / {row[2]} / {row[3]} / {row[4]} / {row[5]} / {row[6]} / {row[7]} / {str(dfEscenaOR.shape)}")

        if row[8] == cont and dfEscena.shape[0] > 0:
            if row[10]:
                fCreaEscenario(Dict, dfEscena, row[1] + '-0', int(row[0]), ascenso=False, porcMin=row[5]/100, porcMax=row[6]/100)
                fCreaEscenario(Dict, dfEscena, row[1] + '-1', int(row[0]), ascenso=True, porcMin=row[5]/100, porcMax=row[6]/100)
            else:
                fCreaEscenario(Dict, dfEscena, row[1], int(row[0]), porcMin=row[5]/100, porcMax=row[6]/100)

            lprint(f"FINAL {str(dfEscena.shape)} / {row[0]} / {row[1]} / {row[2]} / {row[3]} / {row[4]} / {row[5]} / {row[6]} / {row[7]} / {row[8]} / {row[9]}")
        EscenaId = row[0]

    lprint('\n Escenarios aprobados: ')
    for e in Dict:
        lprint(str(Dict[e]['Filtro']) +' '+ str(Dict[e]['dfEscenario'].shape))
    lprint("FIN de creación de Escenario \n")
    return Dict


# #### Configuración de los escenarios seleccionados
def fFiltroEscenario(dfFiltra, Campo, Operador, Valor):
    lprint(str(Campo) + str(Operador) + str(Valor))
    try:
        Valor = int(Valor)
    except ValueError:
        pass

    if type(Valor) == str:
        switcher = {
            '==': dfFiltra[dfFiltra[Campo] == Valor],
            '!=': dfFiltra[dfFiltra[Campo] != Valor],
            'in': dfFiltra[dfFiltra[Campo].astype(str).isin(list(Valor.split(',')))],
            'notin': dfFiltra[~dfFiltra[Campo].astype(str).isin(list(Valor.split(',')))],
            'True': dfFiltra[dfFiltra[Campo] == True],
            'False': dfFiltra[dfFiltra[Campo] == False],
            'like': dfFiltra[dfFiltra[Campo].astype(str).str.contains(Valor, case=False, na=False)],
            'notlike': dfFiltra[~dfFiltra[Campo].astype(str).str.contains(Valor, case=False, na=False)],
            'null': dfFiltra[dfFiltra[Campo].isnull()],
            'notnull': dfFiltra[dfFiltra[Campo].notnull()]
        }

    if type(Valor) == int:
        switcher = {
            '==': dfFiltra[dfFiltra[Campo] == Valor],
            '!=': dfFiltra[dfFiltra[Campo] != Valor],
            '<': dfFiltra[dfFiltra[Campo] < Valor],
            '<=': dfFiltra[dfFiltra[Campo] <= Valor],
            '>': dfFiltra[dfFiltra[Campo] > Valor],
            '>=': dfFiltra[dfFiltra[Campo] >= Valor]
        }
    return switcher.get(Operador, None)


# Crear Escenario
def fCreaEscenario(Dict, miDataFrame, Filtro, id_esc, ascenso=None, porcMin=0, porcMax=1):
    Escena = {}
    # Filtrar por ascenso si corresponde
    if ascenso is None:
        dfEscenario = miDataFrame
    else:
        dfEscenario = miDataFrame[miDataFrame['concurso_ascenso'] == ascenso]

    if not ascenso or ascenso is None:
        dfEscenario = dfEscenario[dfEscenario['mun_inscritos'].between( dfEscenario['mun_inscritos'].quantile(porcMin),  dfEscenario['mun_inscritos'].quantile(porcMax), inclusive="both")]

    dfEscenario = dfEscenario.reset_index(drop=True)

    # Solo continuar si hay más de 4 registros
    if dfEscenario.shape[0] > 4:
        Escena['dfEscenario'] = dfEscenario
        Escena['Filtro'] = Filtro
        Escena['Ascenso'] = ascenso
        Escena['Id'] = id_esc
        Dict['Escena' + str(id_esc) + (f"-{int(ascenso)}" if ascenso is not None else '')] = Escena


# Ejecutar entrenamiento
def fEntrenamiento(dfEmpleo, kfold=False):
  lprint("Inicio - Realizar entrenamiento de escenarios")
  ml = openCosteo()[1]
  boolEntrena = fConsultaScript(openCosteo, f"select valor from {ml}.np_parametros where tipo = 'entrenamiento'").loc[0,'valor']

  if eval(boolEntrena):
    Dict = fEscenarios(dfEmpleo)
    query = f"""
        SELECT id, neuronas, learnrate, dropout, epocas, batch, nombre
        FROM {ml}.np_redneuronal
        WHERE activo
        ORDER BY id
    """
    dfRedNeuronal = fConsultaScript(openCosteo, query)
    listaRed = []
    for i, row in dfRedNeuronal.iterrows():
        red = [int(row[0]), int(row[1]), row[2], row[3], int(row[4]), int(row[5]), row[6]]
        listaRed.append(red)

    fDiccionario(Dict, kfold)
    fAjustarModelo(Dict, listaRed, kfold)

    with open('config/Dict.pkl', 'wb') as fileP:
        pickle.dump(Dict, fileP)
  else:
    with open('config/Dict.pkl', 'rb') as fileP:
        Dict = pickle.load(fileP)
  lprint("FIN de entrenamiento de escenarios \n")
  return Dict


def fDiccionario(Dict, es_kfold=False, n_splits=5):
    lprint("Inicia fDiccionario()")
    columnas = fVariables()['nombre'].to_list()

    for e in Dict:
        df = Dict[e]['dfEscenario'].copy()

        # Seleccionar las columnas necesarias
        Dict[e]['dfModelo'] = df[columnas + ['mun_inscritos']]
        Dict[e]['DatosX'] = fSeleccionarColumnas(df[columnas])
        Dict[e]['DatosY_real'] = df['mun_inscritos'].values.reshape(-1, 1)

        Dict[e]['DatosY'] = np.log1p(df['mun_inscritos']).values.reshape(-1, 1)
#        Dict[e]['DatosY'] = df['mun_inscritos'].values.reshape(-1, 1)

        Dict[e]['entradas'] = Dict[e]['DatosX'].shape[1]

        # Guardar los objetos StandardScaler para usarlos en test y predicción
        Dict[e]['scaler_X'] =  StandardScaler()
#        Dict[e]['scaler_X'] =  RobustScaler()
        Dict[e]['scaler_Y'] =  StandardScaler()

        Dict[e]['X'] = Dict[e]['scaler_X'].fit_transform(Dict[e]['DatosX'])
        Dict[e]['Y'] = Dict[e]['scaler_Y'].fit_transform(Dict[e]['DatosY'])

        # Guardar los escaladores para usarlos en test y predicción
        if es_kfold:
            # Validación Cruzada K-Fold
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            Dict[e]['KFold'] = kf.split(Dict[e]['X'])
        else:
            # División normal en entrenamiento y prueba
            Dict[e]['train_data'], Dict[e]['test_data'], Dict[e]['train_targets'], Dict[e]['test_targets'] = train_test_split(
                Dict[e]['X'], Dict[e]['Y'], test_size=0.2, random_state=1220)
            lprint(f"{str(Dict[e]['train_data'].shape)} / {str(Dict[e]['test_data'].shape)} / {str(Dict[e]['train_targets'].shape)} / {str(Dict[e]['test_targets'].shape)}")

        Dict[e]['Porc_RM'] = 0
        Dict[e]['Porc_Escrit'] = 0
        if Dict[e]['dfEscenario']['mun_inscritos'].sum() != 0:
            Dict[e]['Porc_RM'] = Dict[e]['dfEscenario']['mun_aprobo_vrm'].sum() / Dict[e]['dfEscenario']['mun_inscritos'].sum()
            Dict[e]['Porc_Escrit'] = Dict[e]['dfEscenario']['mun_aprobo_escritas'].sum() / Dict[e]['dfEscenario']['mun_inscritos'].sum()
    lprint("Fin fDiccionario()")


# Conversión de columnas de datos categoricos a columnas OneHot
def fSeleccionarColumnas(dfEscenario):
    df = pd.DataFrame()
    for i, row in fVariables().iterrows():
#        print(row[0], row[1])
        if row[1] in ['num','float']:
            df[row[0]] = dfEscenario[row[0]]
        if row[1] in ['bool']:
            dfCols = pd.get_dummies(dfEscenario[[row[0]]], dtype=float)
            df = pd.concat([df, dfCols], axis=1)
        if row[1] in ['str']:
            dfEscenario[row[0]] = dfEscenario[row[0]].astype(str)
            dfEscenario[row[0]] = dfEscenario[[row[0]]].applymap(rRetirarTildes)

            dfCols = pd.get_dummies(dfEscenario[[row[0]]], dtype=float)
            df[row[0]] = dfEscenario[row[0]].isna().astype(int)
            df = pd.concat([df, dfCols], axis=1)
        if row[1] in ['list']:
            dfTemp = dfEscenario[[row[0]]].copy()
            dfTemp['index'] = dfEscenario.index
            dfTemp[row[0]] = dfTemp[row[0]].apply(lambda x: ast.literal_eval(x)[0] if pd.notna(x) else [])
            dfTemp = dfTemp.explode(row[0])

            dfdummy = pd.get_dummies(dfTemp[row[0]], prefix=row[0])
            dfdummy = pd.concat([dfTemp['index'], dfdummy], axis=1)
            dfdummy = dfdummy.groupby('index').sum().reset_index()
            df[row[0]] = dfEscenario[row[0]].isna().astype(int)
            df = pd.concat([df, dfdummy.drop(columns=['index'])], axis=1)
    return df


def fAjustarModelo(Dict, listaRed, es_kfold=False):
    configurar_mlflow()
    cons = 0  # Contador para identificar modelos
    for escenario, datos in Dict.items():
        historico_modelos = []
        for red in listaRed:
            modelo = fRegresion(datos['entradas'], red[2], red[3])
            detener = EarlyStopping(
                monitor='val_loss',
                patience=max(20, red[4] // 10),
                restore_best_weights=True,
                mode='min',
                min_delta=0.0001
            )
            aprenda = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  
                patience=max(20, red[4] // 20),
                min_lr=1e-6,
                verbose=1
            )

            checkpoint = ModelCheckpoint(
                f'Mejor escenario: {escenario}_red_{red[0]}.keras',
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            )

            if es_kfold:
                val_scores = []
                for train_idx, val_idx in datos['KFold']:
                    # Data augmentation en cada fold (ejemplo para datos numéricos)
                    X_train, y_train = datos['X'][train_idx], datos['Y'][train_idx]
                    X_val, y_val = datos['X'][val_idx], datos['Y'][val_idx]
                    
                    # Añadir ruido gaussiano solo a los datos de entrenamiento
                    X_train_noisy = X_train + np.random.normal(0, 0.01, size=X_train.shape)

                    historial = modelo.fit(
                        X_train_noisy, y_train,
                        epochs=red[4],
                        batch_size=red[5],
                        validation_data=(X_val, y_val),
                        verbose=0,
                        callbacks=[detener, aprenda, checkpoint]
                    )
                    val_scores.append(min(historial.history['val_loss']))
            else:
                X_train_noisy = datos['train_data'] + np.random.normal(0, 0.005, size=datos['train_data'].shape)
                # Entrenar modelo
                historial = modelo.fit(
                    X_train_noisy,
                    datos['train_targets'],
                    epochs=int(red[4]),
                    batch_size=int(red[5]),
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[detener, aprenda, checkpoint]
                )
                val_scores = [min(historial.history['val_loss'])]

            # Procesar resultados del entrenamiento
            historico = {
                'red': red,
                'model': modelo,
                'hist': historial,
                'mae': historial.history['mae'],
                'vmae': historial.history['val_mae'],
                'loss': historial.history['loss'],
                'vloss': historial.history['val_loss'],
                'mae_mean': np.mean(historial.history['mae']),
                'vmae_mean': np.mean(historial.history['val_mae']),
                'loss_mean': np.mean(historial.history['loss']),
                'vloss_mean': np.mean(historial.history['val_loss']),
                'best_val_loss': np.min(historial.history['val_loss']),
                'best_val_mae': np.min(historial.history['val_mae']),
                'Id': cons
            }

            historico_modelos.append(historico)
            cons += 1
            lprint(f"Escenario {datos['Filtro']} / Modelo {cons} ajustado")

            with mlflow.start_run(run_name=f"{escenario}_red_{red[0]}"):
                mlflow.log_param("Escenario", datos['Filtro'])
                mlflow.log_param("Entradas",datos['entradas'])
                mlflow.log_param("neuronas", red[1])
                mlflow.log_param("learnrate", red[2])
                mlflow.log_param("dropout", red[3])
                mlflow.log_param("epocas", red[4])
                mlflow.log_param("batch_size", red[5])
                
                # Entrena el modelo como ya lo haces...
                for epoch, (mae, val_mae, loss, val_loss) in enumerate(
                    zip(historial.history['mae'], historial.history['val_mae'], historial.history['loss'], historial.history['val_loss']) ):
                    mlflow.log_metric("mae", mae, step=epoch)
                    mlflow.log_metric("val_mae", val_mae, step=epoch)
                    mlflow.log_metric("loss", loss, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)

                #dfExample = np.zeros((1, datos['X'].shape[1])) 
                #dfExample = datos['X'][:1].tolist()
                #dfExample = pd.DataFrame( datos['X'][:1], columns=[f"col{i}" for i in range(datos['X'].shape[1])] )
                signature = infer_signature(datos['X'], datos['Y'])
                #mlflow.keras.log_model( modelo, artifact_path="modelo", input_example=dfExample, signature=signature)
                mlflow.keras.log_model( modelo, name="modelo", signature=signature)

        # Guardar el historial completo en el escenario
        datos['Historico'] = historico_modelos


# Construcción del modelo de regresión de la red neuronal
def fRegresion(entradas, rate, out):
    BinNeuronas = rTecho(entradas)
    regularizador = l1_l2(l1=0.00001, l2=0.0001)
    
    model = Sequential([
        Dense(BinNeuronas *4, activation='relu', input_shape=(entradas,), kernel_regularizer=regularizador),
        BatchNormalization(),
        Dropout(out),
        Dense(BinNeuronas *2, activation='relu', kernel_regularizer=regularizador),
        Dropout(out / 2),
        Dense(max(entradas,BinNeuronas //2), activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(
        optimizer=Adam(learning_rate=rate),
        loss='mean_squared_error',
        metrics=['mae'])
    return model


# #### Función general de la proyección

# Predicción de inscritos para los modelos
def fProyeccion(Dict, dfCostos):
    dfVariables = fVariables()
    fProyeccionModelo(Dict, dfCostos)
    Convocatoria = fCompletarConvocatoria(Dict, dfCostos, dfVariables)
    lprint("FIN - Proyección de inscritos finalizada \n")
    return dfCostos, Convocatoria


def fProyeccionModelo(Dict, dfCostos):
    dfCostos = fAgregarSmmlvAgno(dfCostos)
    for e in Dict:
        listaModelo = Dict[e]['dfModelo'].columns.tolist()
        listaModelo.remove('mun_inscritos')
        dfCostosOH = fSeleccionarColumnas(dfCostos[listaModelo])

        ColsCostosElimina = [x for x in dfCostosOH.columns.tolist() if x not in Dict[e]['DatosX'].columns.tolist()]
        if ColsCostosElimina:
            lprint(f"Eliminar columnas adicionales: {ColsCostosElimina}")
            dfCostosOH = fUnificarValoresColumna(dfCostosOH, ColsCostosElimina)
            dfCostosOH.drop(columns=ColsCostosElimina, axis=1, inplace=True)

        ColsCostosAgregar = [x for x in Dict[e]['DatosX'].columns.tolist() if x not in dfCostosOH.columns.tolist()]
        if ColsCostosAgregar:
            lprint(f"Agregar columnas faltantes: {ColsCostosAgregar}")
            for col in ColsCostosAgregar:
                dfCostosOH[col] = 0

      # Asegurar el orden de columnas
        dfCostosOH = dfCostosOH[Dict[e]['DatosX'].columns]

        # Aplicar la misma normalización usada en entrenamiento
        Dict[e]['dfCostosOH'] = Dict[e]['scaler_X'].transform(dfCostosOH)

        for hist in Dict[e]['Historico']:
            predicciones = hist['model'].predict(Dict[e]['dfCostosOH'])
            # Revertir la transformación logarítmica si se aplicó en el entrenamiento
            predicciones = np.asarray(predicciones).reshape(-1, 1)
            predicciones = Dict[e]['scaler_Y'].inverse_transform(predicciones)
            predicciones = np.expm1(predicciones).flatten()
            hist['Resultado'] = np.clip(predicciones, 0, Dict[e]['DatosY_real'].max() * 1.5)

            lprint(f"Proyección realizada para el escenario {e}")


def fUnificarValoresColumna(dfOH, ColsCostosElimina):
    for i, row in fVariables().iterrows():
        if row[1] in ['str','list']:
            dfOH[row[0]] = dfOH[row[0]].isna().astype(int)
            for col in dfOH[ColsCostosElimina]:
                if row[0] in col:
                    dfOH[row[0]] = dfOH[row[0]] | dfOH[col].isna().astype(int)
    return dfOH


# Completar la convocatoria con los resultados de cada empleo
def fCompletarConvocatoria(Dict, dfCostos, dfVariables):
    Convocatoria = []
    for i in range(dfCostos.shape[0]):
        Empleo = {}
        Empleo['Indice'] = i
        Empleo['Id'] = dfCostos.iloc[i]['id']
        Empleo['Empleo_Id'] = dfCostos.iloc[i]['empleo_id']
        for k, row in dfVariables.iterrows():
            Empleo[row[0]] = dfCostos.iloc[i][row[0]]
        for e in Dict:
            f = Dict[e]['Filtro']
            Empleo[f] = {}
            Empleo[f]['Neuronas'] = {}
            for hist in Dict[e]['Historico']:
                if 'vacantes' in Empleo:
                    Empleo[f]['Neuronas'][hist['red'][0]] = int(max(Empleo['vacantes'], hist['Resultado'][i]))
                else:
                    Empleo[f]['Neuronas'][hist['red'][0]] = int(hist['Resultado'][i])
        Convocatoria.append(Empleo)
    return Convocatoria
