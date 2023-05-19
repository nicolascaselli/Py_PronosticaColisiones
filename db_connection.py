import psycopg2
import json
from logger import setup_logger

logger = setup_logger("AlertaColisionesAeropuerto")
def create_connection():
    try:
        with open('config/db_config.json') as config_file:
            config = json.load(config_file)

        connection = psycopg2.connect(
            database=config['database'],
            user=config['user'],
            password=config['password'],
            host=config['host'],
            port=config['port']
        )
        return connection
    except (Exception, psycopg2.Error) as error:
        print("Error al conectar a la base de datos:", error)
        logger.debug(f'Error al conectar a la base de datos: {error}')
        return None
