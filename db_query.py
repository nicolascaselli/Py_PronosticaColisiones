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

def extraeDatosEntrenamiento():
    # Establecer la conexión con la base de datos
    conn = create_connection()

    # Crear un cursor para ejecutar la consulta
    cursor = conn.cursor()

    # Ejecutar el procedimiento almacenado
    cursor.callproc("extraeDatosEntrenamiento")

    # Obtener el resultado del procedimiento almacenado
    resultado = cursor.fetchall()

    # Cerrar el cursor y la conexión
    cursor.close()
    conn.close()

    # Separar las columnas en listas
    X_Train = [fila[:14] for fila in resultado]
    Y_Train = [fila[14] for fila in resultado]

    return X_Train, Y_Train

def extraeDatosPronostico():
    # Establecer la conexión con la base de datos
    conn = create_connection()

    # Crear un cursor para ejecutar la consulta
    cursor = conn.cursor()

    # Ejecutar el procedimiento almacenado
    cursor.callproc("extraeDatosPronostico")

    # Obtener el resultado del procedimiento almacenado
    X_Test = cursor.fetchall()

    # Cerrar el cursor y la conexión
    cursor.close()
    conn.close()

    return X_Test
