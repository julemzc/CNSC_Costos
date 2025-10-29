import logging
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
from main import pipeline

# Configuración del logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app)
CORS(app)


class ConsumeApi(Resource):
    print("llegue ala variable consume api: " + str(Resource))
    def __init__(self):
        logger.info("Instanciando ConsumeApi")

    def get(self, id):
        logger.info(f"GET request received for id: {id}")

        # Validación del ID (opcional)
        if not id.isdigit():
            logger.warning("Invalid ID format: %s", id)
            return {"error": "Invalid ID format"}, 400

        try:
            result = pipeline(id)  # Suponiendo que pipeline retorna algo
            logger.info(f"Pipeline executed successfully for id: {id}")
            return {"estado": "ok"}, 200
        except Exception as error:
            logger.error(f"Error processing id {id}: {error}", exc_info=True)
            return {"error": "internal error, verify input params"}, 500


# Crear rutas
logger.info("Configurando ruta para ConsumeApi")
api.add_resource(ConsumeApi, '/api/proyectar/<id>')

# Ejecutar la aplicación
if __name__ == '__main__':
    logger.info("Iniciando la aplicación")
    app.run(host='0.0.0.0', port=8182, debug=True)
