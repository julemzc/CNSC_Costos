python -m venv .venv
.venv\Scripts\activate


.venv\Scripts\python.exe -m pip install --upgrade pip 
pip install pandas psycopg2 sqlalchemy unidecode mlflow keras tensorflow seaborn statsmodels xlsxwriter
pip install streamlit, flask_restful flask_cors geopy folium

pip freeze > requeriments.txt

pip install -r requeriments.txt

#Para trabajar con notebook
.venv\Scripts\activate
pip install jupyter ipykernel
python -m ipykernel install --user --name=costos_env --display-name "Costos env"


# para ejecutar streamlit
cd Costos
.venv\Scripts\activate

streamlit run main_streamlit.py