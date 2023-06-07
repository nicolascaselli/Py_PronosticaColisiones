from logger import setup_logger
from componenteIA import *
import db_query as dbq
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

    # Ejecutar el módulo
    # run()

    # Ejemplo de uso
    #X_train, Y_train = dbq.extraeDatosPronostico()  # Datos de salida para entrenamiento (etiquetas)
    #print(f"Datos de entrenamiento:\n {X_train}")
    #print(f"Etiquetas de entrenamiento:\n {Y_train}")
    componenteIA = Pronosticador()
    componenteIA.run()
    # Llamada a la función de aprendizaje incremental
    # aprendizaje_incremental(X_train, Y_train, ruta_modelo)

    logger.debug('Finalizando sistema de pronóstico de alertas')


