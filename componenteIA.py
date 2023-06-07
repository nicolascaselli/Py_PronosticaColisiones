import time
import tensorflow as tf
import numpy as np
import configparser
import db_query as dbq
import os
from sklearn.model_selection import train_test_split

class Pronosticador():

    def __init__(self):
        self.ruta_configuracion = os.path.join('config','config.ini')  # Ruta del archivo de configuración
        # Carga de la configuración
        self.config = configparser.ConfigParser()
        self.config.read(self.ruta_configuracion)
        self.std = None
        self.mean = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.loss = None
        self.accuracy = None
        self.pronostico_activo = self.config.getboolean("APP", "PronosticoActivo")
        self.SegundosIntervaloPronostico = int(self.config.get("APP", "SegundosIntervaloPronostico"))
        ruta_modelo = os.path.join(self.config.get("APP", "CarpetaModelo"), self.config.get("APP",
                                                                                            "NombreModelo"))  # Ruta donde se encuentra el modelo entrenado
        # Carga del modelo o lo crea si no existe
        self.modelo_entrenado = self.cargar_modelo(ruta_modelo)



    def crear_nuevo_modelo(self):

        X_train, Y_train = dbq.extraeDatosEntrenamiento() #consultar a la BD los datos de entrenamiento y sus etiquetas
        X_train = np.where(X_train == 0, 0.01, X_train) #Reemplazamos los ceros por 0.01
        input_dim = X_train.shape[1]
        output_dim = Y_train.shape[1] if len(Y_train.shape) > 1 else 1  # Obtener la dimensión de las etiquetas de salida
        # print(f"PREVIO A LA NORMALIZACION\nX_train.shape[0]: {X_train.shape[0]} X_train.shape[1]: {X_train.shape[1]}")
        # print(f"Y_train.shape[0]: {Y_train.shape[0]} ")

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
        # print(f"POST A LA NORMALIZACION\nX_train.shape[0]: {X_train.shape[0]} X_train.shape[1]: {X_train.shape[1]}")
        # print(f"Y_train.shape[0]: {Y_train.shape[0]} ")
        # print(f"X_test.shape[0]: {X_test.shape[0]} X_test.shape[1]: {X_test .shape[1]}")
        # print(f"Y_train.shape[0]: {Y_test.shape[0]} ")

        # Normalización de datos
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        self.std = std # guardamos la desviación estándar
        self.mean = mean  # guardamos el promedio
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test= Y_test
        self.std.tofile(os.path.join(self.config.get('APP', 'CarpetaModelo'), 'std.dat'))
        self.mean.tofile(os.path.join(self.config.get('APP', 'CarpetaModelo'), 'mean.dat'))


        # Definir la arquitectura del modelo (red neuronal)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(output_dim, activation='sigmoid'))

        # Compilar el modelo
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        # Entrenar el modelo
        model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test))
        # Evaluar el modelo en el conjunto de prueba
        loss, accuracy = model.evaluate(X_test, Y_test)
        self.loss = loss
        self.accuracy = accuracy
        print(f'Loss: {loss}, Accuracy: {accuracy}')

        return model
    def ajustar_modelo(self, model, datos_entrenamiento):
        # Ajusta el modelo utilizando los nuevos datos de entrenamiento
        X_train = datos_entrenamiento["X_train"]
        Y_train = datos_entrenamiento["Y_train"]
        model.fit(X_train, Y_train, epochs=1000, batch_size=32)
        return model

    def obtener_datos_entrenamiento_previos(self):
        # Lógica para obtener los datos de entrenamiento previos
        datos_entrenamiento_previos = ...
        return datos_entrenamiento_previos

    def combinar_datos(self, datos_previos, datos_nuevos):
        # Lógica para combinar los datos previos con los nuevos datos
        datos_combinados = ...
        return datos_combinados

    def obtener_datos_pronostico(self):
        # Lógica para obtener los nuevos datos de pronóstico
        datos_pronostico = dbq.extraeDatosEntrenamiento()
        return datos_pronostico
    def aprendizaje_incremental(self, ruta_modelo, datos_nuevos):
        # Carga del modelo
        modelo_entrenado = self.cargar_modelo(ruta_modelo)

        # Obtener datos de entrenamiento previos
        datos_entrenamiento_previos = self.obtener_datos_entrenamiento_previos()

        # Combinar datos previos con los nuevos datos
        datos_entrenamiento_actualizados = self.combinar_datos(datos_entrenamiento_previos, datos_nuevos)

        # Ajustar el modelo con los nuevos datos utilizando model.fit()
        X_train_actualizado = datos_entrenamiento_actualizados["X_train"]
        Y_train_actualizado = datos_entrenamiento_actualizados["Y_train"]
        modelo_entrenado.fit(X_train_actualizado, Y_train_actualizado)

        # Obtener nuevos datos para el pronóstico
        datos_pronostico = self.obtener_datos_pronostico()

        # Realizar el pronóstico
        resultado_pronostico = self.realizar_pronostico(modelo_entrenado, datos_pronostico)

        # Obtener etiquetas reales correspondientes a los datos de pronóstico
        etiquetas_reales = self.obtener_etiquetas_reales()

        # Calcular el porcentaje de error
        porcentaje_error = self.calcular_error(resultado_pronostico, etiquetas_reales)

        # Imprimir resultados
        print("Pronóstico:", resultado_pronostico)
        print("Porcentaje de error:", porcentaje_error)

        # Guardar el modelo actualizado
        self.guardar_modelo(modelo_entrenado, ruta_modelo)


    def cargar_modelo(self, ruta_modelo):
        # Carga del modelo entrenado desde el archivo
        try:
            model = tf.keras.models.load_model(ruta_modelo)
            self.mean = np.fromfile(os.path.join(self.config.get('APP', 'CarpetaModelo'), 'mean.dat'))
            self.std = np.fromfile(os.path.join(self.config.get('APP', 'CarpetaModelo'), 'std.dat'))
            print("Modelo cargado exitosamente")
        except:
            print("Modelo no encontrado. Se creará un modelo nuevo.")
            model = self.crear_nuevo_modelo()
            self.guardar_modelo(model, ruta_modelo)
        return model

    def guardar_modelo(self, modelo, ruta):
        modelo.save(ruta)
        print("El modelo se ha guardado correctamente en:", ruta)
    def realizar_pronostico(self, modelo, X_pronostico, mean, std):
        # Realizar predicciones en nuevos datos
        datos = (X_pronostico - mean) / std  # Normalizar los nuevos datos
        pronostico = modelo.predict(datos)
        return pronostico

    def calcular_error(self, pronostico, etiquetas):
        # Calcula el porcentaje de error entre el pronóstico y las etiquetas reales
        error = np.abs(pronostico - etiquetas) / etiquetas
        porcentaje_error = np.mean(error) * 100
        return porcentaje_error

    def run(self):

        while self.pronostico_activo:
            # Obtener nuevos datos para el pronóstico
            X_pronostico, Y_pronostico = self.obtener_datos_nuevos()  # Función para obtener los datos nuevos
            if len(X_pronostico)>=1:
                # Realizar el pronóstico
                resultado_pronostico = self.realizar_pronostico(self.modelo_entrenado, X_pronostico, mean=self.mean, std=self.std)

                # Obtener la clase con la mayor probabilidad
                clase_predicha = np.argmax(resultado_pronostico, axis=1)

                # Obtener la probabilidad más alta
                certeza = np.max(resultado_pronostico, axis=1)

                # Imprimir resultados
                print("Pronóstico:", resultado_pronostico)
                print(f"Clase predicha: {clase_predicha}\nCerteza: {certeza} ")
            else:
                print("Sin datos para pronosticar...")

            time.sleep(self.SegundosIntervaloPronostico)  # Esperar N segundo antes de realizar el siguiente pronóstico

            # Actualizar el valor de pronostico_activo desde el archivo de configuración
            pronostico_activo = self.config.getboolean("APP", "PronosticoActivo")

    # Funciones para obtener datos nuevos y etiquetas reales
    def obtener_datos_nuevos(self):
        # Lógica para obtener los datos nuevos
        X_pronostico, Y_pronostico = dbq.extraeDatosPronostico()
        return X_pronostico, Y_pronostico

    def obtener_etiquetas_reales(self):
        # Lógica para obtener las etiquetas reales correspondientes a los datos nuevos
        etiquetas_reales = ...
        return etiquetas_reales