# CodEx - ETL Pipeline para Mercado Libre
Este proyecto contiene un pipeline de ETL diseñado para ejecutar la red neuronal.

### Estructura del Proyecto
- `config/`: Contiene el archivo `config.yaml` con las configuraciones de red + dos .pkl que leen el Diccionario de datos
- `logs/`: Directorio donde se almacenan los logs de la ejecución.
- `mlruns/`: Directorio que contiene las metricas de las ejecuciones.
- `output/`: Directorio donde se exportan los resultados de la ejecución.
- `sql/`: Consultas a la base de datos de SIMO.
- `src/`: Contiene los scripts de código fuente y funciones utilitarias.

### Configuración
Las configuraciones de red en el archivo `config/config.yaml`.


### Para crear y activar un ambiente virtual
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Ejecución desde la aplicación
Para ejecutar el pipeline,  ejecutar el siguiente comando:
```bash
python src/main.py
```

### Ejecución con streamlit para crear y seleccionar las convocatorias creadas
Abrir el navegador para ejecutar la red neuronal
```bash
cd Costos
.venv\Scripts\activate
streamlit run main_streamlit.py
```

### Para trabajar con notebook
Habilitar el ambiente creado en notebook
```bash
.venv\Scripts\activate
pip install jupyter ipykernel
python -m ipykernel install --user --name=costos_env --display-name "Costos env"
```

### mlflow para visualizar las metricas
Abrir el navegador con la siguiente url `http://127.0.0.1:5000`
```bash
mlflow ui --backend-store-uri file:///C:/Users/jlmartinez/Costos/mlruns
```