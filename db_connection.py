import configparser
import psycopg2
import json
import os
from logger import setup_logger

logger = setup_logger("AlertaColisionesAeropuerto")
def create_connection():
    try:
        # Carga de la configuración
        config = configparser.ConfigParser()
        config.read(os.path.join('config','config.ini'))

        connection = psycopg2.connect(
            database=config.get('BD','database'),
            user=config.get('BD','user'),
            password=config.get('BD','password'),
            host=config.get('BD','host'),
            port=config.get('BD','port')
        )
        return connection
    except (Exception, psycopg2.Error) as error:
        print("Error al conectar a la base de datos:", error)
        logger.debug(f'Error al conectar a la base de datos: {error}')
        return None
