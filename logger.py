import logging
import os
from datetime import datetime


def setup_logger(log_file):
    # Obtener la fecha actual
    current_date = datetime.now().strftime("%Y%m%d")
    # Crear la carpeta "logs" si no existe
    if not os.path.exists("logs"):
        os.makedirs("logs")
    # Configurar el logger
    logger = logging.getLogger('AlertaColisiones')
    logger.setLevel(logging.DEBUG)

    # Crear un formateador para el log
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Crear un manejador para escribir el log en un archivo
    log_file = os.path.join("logs", f"{log_file}_{current_date}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Agregar el manejador al logger
    logger.addHandler(file_handler)
    return logger