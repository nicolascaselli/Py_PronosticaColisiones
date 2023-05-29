import time
import tensorflow as tf
import numpy as np
import configparser
import db_query as dbq

def crear_nuevo_modelo():
    X_train, Y_train = dbq.extraeDatosEntrenamiento() #consultar a la BD los datos de entrenamiento y sus etiquetas
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]

    # Definir la arquitectura del modelo
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(output_dim, activation='sigmoid'))

    # Compilar el modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(X_train, Y_train, epochs=1000, batch_size=32)

    return model
def ajustar_modelo(model, datos_entrenamiento):
    # Ajusta el modelo utilizando los nuevos datos de entrenamiento
    X_train = datos_entrenamiento["X_train"]
    Y_train = datos_entrenamiento["Y_train"]
    model.fit(X_train, Y_train, epochs=1000, batch_size=32)
    return model

def obtener_datos_entrenamiento_previos():
    # Lógica para obtener los datos de entrenamiento previos
    datos_entrenamiento_previos = ...
    return datos_entrenamiento_previos

def combinar_datos(datos_previos, datos_nuevos):
    # Lógica para combinar los datos previos con los nuevos datos
    datos_combinados = ...
    return datos_combinados

def obtener_datos_pronostico():
    # Lógica para obtener los nuevos datos de pronóstico
    datos_pronostico = dbq.extraeDatosEntrenamiento()
    return datos_pronostico
def aprendizaje_incremental(ruta_modelo, datos_nuevos):
    # Carga del modelo
    modelo_entrenado = cargar_modelo(ruta_modelo)

    # Obtener datos de entrenamiento previos
    datos_entrenamiento_previos = obtener_datos_entrenamiento_previos()

    # Combinar datos previos con los nuevos datos
    datos_entrenamiento_actualizados = combinar_datos(datos_entrenamiento_previos, datos_nuevos)

    # Ajustar el modelo con los nuevos datos utilizando model.fit()
    X_train_actualizado = datos_entrenamiento_actualizados["X_train"]
    Y_train_actualizado = datos_entrenamiento_actualizados["Y_train"]
    modelo_entrenado.fit(X_train_actualizado, Y_train_actualizado)

    # Obtener nuevos datos para el pronóstico
    datos_pronostico = obtener_datos_pronostico()

    # Realizar el pronóstico
    resultado_pronostico = realizar_pronostico(modelo_entrenado, datos_pronostico)

    # Obtener etiquetas reales correspondientes a los datos de pronóstico
    etiquetas_reales = obtener_etiquetas_reales()

    # Calcular el porcentaje de error
    porcentaje_error = calcular_error(resultado_pronostico, etiquetas_reales)

    # Imprimir resultados
    print("Pronóstico:", resultado_pronostico)
    print("Porcentaje de error:", porcentaje_error)

    # Guardar el modelo actualizado
    guardar_modelo(modelo_entrenado, ruta_modelo)


def cargar_modelo(ruta_modelo):
    # Carga del modelo entrenado desde el archivo
    try:
        model = tf.keras.models.load_model(ruta_modelo)
        print("Modelo cargado exitosamente")
    except:
        print("Modelo no encontrado. Se creará un modelo nuevo.")
        model = crear_nuevo_modelo()
    return model

def guardar_modelo(modelo, ruta):
    modelo.save(ruta)
    print("El modelo se ha guardado correctamente en:", ruta)
def realizar_pronostico(modelo, datos):
    # Realiza el pronóstico utilizando el modelo cargado
    pronostico = modelo.predict(datos)
    return pronostico

def calcular_error(pronostico, etiquetas):
    # Calcula el porcentaje de error entre el pronóstico y las etiquetas reales
    error = np.abs(pronostico - etiquetas) / etiquetas
    porcentaje_error = np.mean(error) * 100
    return porcentaje_error

def run():

    ruta_configuracion = "config\config.ini"  # Ruta del archivo de configuración

    # Carga de la configuración
    config = configparser.ConfigParser()
    config.read(ruta_configuracion)
    pronostico_activo = config.getboolean("Config", "PronosticoActivo")

    ruta_modelo = config.get("Config", "RutaNombreModelo")  # Ruta donde se encuentra el modelo entrenado

    # Carga del modelo
    modelo_entrenado = cargar_modelo(ruta_modelo)

    while pronostico_activo:
        # Obtener nuevos datos para el pronóstico
        datos_nuevos = obtener_datos_nuevos()  # Función para obtener los datos nuevos

        # Realizar el pronóstico
        resultado_pronostico = realizar_pronostico(modelo_entrenado, datos_nuevos)

        # Obtener la clase con la mayor probabilidad
        clase_predicha = np.argmax(resultado_pronostico, axis=1)

        # Obtener la probabilidad más alta
        certeza = np.max(resultado_pronostico, axis=1)

        # Imprimir resultados
        print("Pronóstico:", resultado_pronostico)
        print(f"Clase predicha: {clase_predicha}\nCerteza: {certeza} ")

        time.sleep(config.get("Config", "SegundosIntervaloPronostico"))  # Esperar N segundo antes de realizar el siguiente pronóstico

        # Actualizar el valor de pronostico_activo desde el archivo de configuración
        pronostico_activo = config.getboolean("Config", "PronosticoActivo")

# Funciones para obtener datos nuevos y etiquetas reales
def obtener_datos_nuevos():
    # Lógica para obtener los datos nuevos
    datos_nuevos = dbq.extraeDatosPronostico()
    return datos_nuevos

def obtener_etiquetas_reales():
    # Lógica para obtener las etiquetas reales correspondientes a los datos nuevos
    etiquetas_reales = ...
    return etiquetas_reales

# Ejecutar el módulo
run()

# Ejemplo de uso
X_train = ...  # Datos de entrada para entrenamiento (características)
Y_train = ...  # Datos de salida para entrenamiento (etiquetas)
ruta_modelo = "modelo.h5"  # Ruta donde se guardará el modelo

# Llamada a la función de aprendizaje incremental
aprendizaje_incremental(X_train, Y_train, ruta_modelo)
