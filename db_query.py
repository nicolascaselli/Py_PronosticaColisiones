import psycopg2
import numpy as np
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
    try:

        # Establecer la conexión con la base de datos
        conn = create_connection()

        # Crear un cursor para ejecutar la consulta
        cursor = conn.cursor()
        tipoRegistro = 1

        # Definir las variables de salida
        cur_datosPronostico = None
        numerror = None
        msjerror = None
        # Ejecutar el procedimiento almacenado
        cursor.execute('call dbsds.sds$obtiene_datosPronostico(%s, %s, %s, %s)', [tipoRegistro, cur_datosPronostico, numerror, msjerror])
        retorno = cursor.fetchall()

        # Obtener los valores de salida
        numerror_value = retorno[0][1]
        msjerror_value = retorno[0][2]

        cursor.execute('FETCH ALL FROM "{0}"'.format(retorno[0][0]))
        # Obtener el cursor ref como resultado
        cur_datosPronostico = cursor.fetchall()

        registros = list()
        for registro in cur_datosPronostico:
            registros.append(list(registro))

        # Cerrar el cursor y la conexión
        cursor.close()
        conn.close()

        # Separar las columnas en listas
        X_Train = [fila[2:13] for fila in cur_datosPronostico] #datos
        Y_Train = [fila[13] for fila in cur_datosPronostico]    #etiquetas

        X_Train = np.array(X_Train)
        Y_Train = np.array(Y_Train)
        return X_Train, Y_Train
    except (Exception, psycopg2.Error) as error:
        print(f'Error al extraeDatosEntrenamiento: {error}')
        logger.debug(f'Error al extraeDatosEntrenamiento: {error}')
        return None


def extraeDatosPronostico():
    try:

        # Establecer la conexión con la base de datos
        conn = create_connection()

        # Crear un cursor para ejecutar la consulta
        cursor = conn.cursor()
        #1 para datos dee tipo reales, 2 para pronóstico
        tipoRegistro = 2

        # Definir las variables de salida
        cur_datosPronostico = None
        numerror = None
        msjerror = None
        # Ejecutar el procedimiento almacenado
        cursor.execute('call dbsds.sds$obtiene_datosPronostico(%s, %s, %s, %s)',
                       [tipoRegistro, cur_datosPronostico, numerror, msjerror])
        retorno = cursor.fetchall()

        # Obtener los valores de salida
        numerror_value = retorno[0][1]
        msjerror_value = retorno[0][2]

        cursor.execute('FETCH ALL FROM "{0}"'.format(retorno[0][0]))
        # Obtener el cursor ref como resultado
        cur_datosPronostico = cursor.fetchall()

        registros = list()
        for registro in cur_datosPronostico:
            registros.append(list(registro))

        # Cerrar el cursor y la conexión
        cursor.close()
        conn.close()

        # Separar las columnas en listas
        X_Train = [fila[2:13] for fila in cur_datosPronostico]  # datos
        Y_Train = [fila[13] for fila in cur_datosPronostico]  # etiquetas

        X_Train = np.array(X_Train)
        Y_Train = np.array(Y_Train)
        return X_Train, Y_Train
    except (Exception, psycopg2.Error) as error:
        print(f'Error al extraeDatosEntrenamiento: {error}')
        logger.debug(f'Error al extraeDatosEntrenamiento: {error}')
        return None