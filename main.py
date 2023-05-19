from logger import setup_logger
from db_query import execute_query


if __name__ == '__main__':
    # Configura el logger con el nombre del archivo de log
    logger = setup_logger('AlertaColisionesAeropuerto')
    # Ejemplo de uso del logger
    '''
    logger.debug('Este es un mensaje de depuración')
    logger.info('Este es un mensaje de información')
    logger.warning('Este es un mensaje de advertencia')
    logger.error('Este es un mensaje de error')
    logger.critical('Este es un mensaje crítico')
    '''
    logger.debug('Iniciando sistema de pronóstico de alertas')

    query = "SELECT * FROM dbsds.parametro"
    result = execute_query(query)

    if result:
        for row in result:
            print(row)
    else:
        print("error al ejecutar la consulta.")

    logger.debug('Finalizando sistema de pronóstico de alertas')


