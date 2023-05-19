import psycopg2

from db_connection import create_connection
from logger import setup_logger

logger = setup_logger("AlertaColisionesAeropuerto")

def execute_query(query):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            connection.close()
            return result
        except (Exception, psycopg2.Error) as error:
            print("Error al ejecutar la consulta:", error)
            logger.debug(f'Error al ejecutar la consulta: {error}')
            return None
    else:
        return None

def insert_record(table, values):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            query = f"INSERT INTO {table} VALUES ({','.join(['%s']*len(values))})"
            cursor.execute(query, values)
            connection.commit()
            cursor.close()
            connection.close()
            print("Registro insertado correctamente.")
        except (Exception, psycopg2.Error) as error:
            print("Error al insertar el registro:", error)
    else:
        print("Error al establecer la conexión con la base de datos.")