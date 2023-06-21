import math

import matplotlib.pyplot as plt
import time
import tensorflow as tf
import numpy as np
import configparser
import db_query as dbq
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
        self.modelo_entrenado = self.cargar_modeloCNN(ruta_modelo)

    def normalize_min_max(self, data):
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data
    def crear_nuevo_modeloCNN(self):

        X_train, Y_train = dbq.extraeDatosEntrenamiento() #consultar a la BD los datos de entrenamiento y sus etiquetas
        X_train = np.where(X_train == 0, 0.01, X_train) #Reemplazamos los ceros por 0.01
        input_dim = X_train.shape[1]
        output_dim = Y_train.shape[1] if len(Y_train.shape) > 1 else 1  # Obtener la dimensión de las etiquetas de salida
        # print(f"PREVIO A LA NORMALIZACION\nX_train.shape[0]: {X_train.shape[0]} X_train.shape[1]: {X_train.shape[1]}")
        # print(f"Y_train.shape[0]: {Y_train.shape[0]} ")

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=38)
        # print(f"POST A LA NORMALIZACION\nX_train.shape[0]: {X_train.shape[0]} X_train.shape[1]: {X_train.shape[1]}")
        # print(f"Y_train.shape[0]: {Y_train.shape[0]} ")
        # print(f"X_test.shape[0]: {X_test.shape[0]} X_test.shape[1]: {X_test .shape[1]}")
        # print(f"Y_train.shape[0]: {Y_test.shape[0]} ")

        # Normalización de datos

        X_train = self.normalize_min_max(X_train)

        X_test = self.normalize_min_max(X_test)
        np.savetxt(os.path.join("model", "X_train.csv"), X_train, delimiter=';' )
        np.savetxt(os.path.join("model","X_test.csv"), X_test, delimiter=';' )
        np.savetxt(os.path.join("model", "Y_train.csv"), Y_train, delimiter=';')
        np.savetxt(os.path.join("model", "Y_test.csv"), Y_test, delimiter=';')

        self.std = np.mean(X_train, axis=0) # guardamos la desviación estándar
        self.mean = np.std(X_train, axis=0)  # guardamos el promedio
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test= Y_test
        self.std.tofile(os.path.join(self.config.get('APP', 'CarpetaModelo'), 'std.dat'))
        self.mean.tofile(os.path.join(self.config.get('APP', 'CarpetaModelo'), 'mean.dat'))


        # Definir la arquitectura del modelo (red neuronal densa)
        # model = tf.keras.models.Sequential()
        # model.add(tf.keras.layers.Dense(11, activation='relu', input_shape=(input_dim,)))
        # model.add(tf.keras.layers.Dense(64, activation='relu'))
        # model.add(tf.keras.layers.Dense(128, activation='relu'))
        # model.add(tf.keras.layers.Dense(32, activation='relu'))
        # model.add(tf.keras.layers.Dense(16, activation='relu'))
        # model.add(tf.keras.layers.Dense(8, activation='relu'))
        # model.add(tf.keras.layers.Dense(output_dim, activation='sigmoid'))

        # Compilar el modelo
        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])
        ## RED NEURONAL CONVOLUCIONAL
        #modificamos las dimensiones para que sean soportadas por CNN
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(128, 3, input_shape=(input_dim,1), activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='same'),  # 2,2 es el tamano de la matriz

            tf.keras.layers.Conv1D(128, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='same'),  # 2,2 es el tamano de la matriz

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compilar el modelo
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        # Entrenar el modelo
        #self.Y_train = self.Y_train.astype(int)
        #self.Y_train = self.Y_train.reshape(self.Y_train.shape[0], 1)

        historial = model.fit(self.X_train, self.Y_train, epochs=10000, batch_size=800, steps_per_epoch=math.ceil(input_dim/32))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(historial.history["loss"])
        plt.show()
        # Evaluar el modelo en el conjunto de prueba
        # loss, accuracy = model.evaluate(self.X_test, self.Y_test)
        # self.loss = loss
        # self.accuracy = accuracy
        # print(f'Loss: {loss}, Accuracy: {accuracy}')

        return model
    def evaluarModeloDatosPruebasCNN(self):
        # Obtener las predicciones del modelo en los datos de prueba

        Y_pred = self.modelo_entrenado.predict(self.X_test)

        # Redondear las predicciones a 0 o 1, si es necesario
        Y_pred_rounded = np.round(Y_pred)

        # Calcular la precisión del modelo en los datos de prueba
        accuracy = np.mean(Y_pred_rounded == self.Y_test)

        # Imprimir la precisión del modelo
        print("Precisión del modelo en los datos de prueba:", accuracy)
        # print(f"etiquetas de Pruebas:\n {self .Y_test}")
        # print(f"etiquetas pronosticadas:\n {Y_pred_rounded}")
        # print(f"etiquetas pronosticadas:\n {Y_pred_rounded==self.Y_test}")
    def ajustar_modeloCNN(self, model, datos_entrenamiento):
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
        modelo_entrenado = self.cargar_modeloCNN(ruta_modelo)

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
        resultado_pronostico = self.realizar_pronosticoCNN(modelo_entrenado, datos_pronostico)

        # Obtener etiquetas reales correspondientes a los datos de pronóstico
        etiquetas_reales = self.obtener_etiquetas_reales()

        # Calcular el porcentaje de error
        porcentaje_error = self.calcular_errorCNN(resultado_pronostico, etiquetas_reales)

        # Imprimir resultados
        print("Pronóstico:", resultado_pronostico)
        print("Porcentaje de error:", porcentaje_error)

        # Guardar el modelo actualizado
        self.guardar_modeloCNN(modelo_entrenado, ruta_modelo)


    def cargar_modeloCNN(self, ruta_modelo):
        # Carga del modelo entrenado desde el archivo
        try:
            model = tf.keras.models.load_model(ruta_modelo)
            self.mean = np.fromfile(os.path.join(self.config.get('APP', 'CarpetaModelo'), 'mean.dat'))
            self.std = np.fromfile(os.path.join(self.config.get('APP', 'CarpetaModelo'), 'std.dat'))
            self.X_test = np.loadtxt(os.path.join("model","X_test.csv"), delimiter=';')
            self.Y_test = np.loadtxt(os.path.join("model", "Y_test.csv"), delimiter=';')
            self.X_train = np.loadtxt(os.path.join("model", "X_train.csv"), delimiter=';')
            self.Y_train = np.loadtxt(os.path.join("model", "Y_train.csv"), delimiter=';')
            print("Modelo cargado exitosamente")
        except:
            print("Modelo no encontrado. Se creará un modelo nuevo.")
            model = self.crear_nuevo_modeloCNN()
            self.guardar_modeloCNN(model, ruta_modelo)
        return model

    def guardar_modeloCNN(self, modelo, ruta):
        modelo.save(ruta)
        print("El modelo se ha guardado correctamente en:", ruta)
    def realizar_pronosticoCNN(self, modelo, X_pronostico):
        # Realizar predicciones en nuevos datos
        datos = self.normalize_min_max(X_pronostico)
        datos = datos.reshape(datos.shape[0], datos.shape[1], 1)
        pronostico = modelo.predict(datos)
        return pronostico

    def calcular_errorCNN(self, pronostico, etiquetas):
        # Calcula el porcentaje de error entre el pronóstico y las etiquetas reales
        error = np.abs(pronostico - etiquetas) / etiquetas
        porcentaje_error = np.mean(error) * 100
        return porcentaje_error
    def calculapronosticoSVM(self):



        '''
        # Crear un clasificador SVM lineal
        svm = SVC(kernel='linear')

        # Crear un clasificador SVM con kernel RBF
        svm_rbf = SVC(kernel='rbf')
        
        # Crear un clasificador SVM con kernel polinomial de grado 3
        svm_poly = SVC(kernel='poly', degree=3)
        
        # Crear un clasificador SVM con kernel sigmoide
        svm_sig = SVC(kernel='sigmoid')
        
        # Crear un clasificador SVM con kernel sigmoidal gaussiano
        svm_gauss = SVC(kernel='sigmoid', gamma='scale')
        '''
        # Crear un clasificador SVM
        svm = SVC(kernel='rbf')

        # Entrenar el modelo
        svm.fit(self.X_train, self.Y_train)

        # Realizar predicciones en los datos de entrenamiento
        y_pred_train = svm.predict(self.X_train)

        # Calcular las métricas de rendimiento en los datos de entrenamiento
        train_accuracy = accuracy_score(self.Y_train, y_pred_train)
        train_precision = precision_score(self.Y_train, y_pred_train)
        train_recall = recall_score(self.Y_train, y_pred_train)
        train_f1 = f1_score(self.Y_train, y_pred_train)

        # Realizar predicciones en los datos de prueba
        y_pred_test = svm.predict(self.X_test)

        # Calcular las métricas de rendimiento en los datos de prueba
        test_accuracy = accuracy_score(self.Y_test, y_pred_test)
        test_precision = precision_score(self.Y_test, y_pred_test)
        test_recall = recall_score(self.Y_test, y_pred_test)
        test_f1 = f1_score(self.Y_test, y_pred_test)

        # Imprimir las métricas de rendimiento
        print("Train Accuracy:", train_accuracy)
        print("Train Precision:", train_precision)
        print("Train Recall:", train_recall)
        print("Train F1 Score:", train_f1)
        print("Test Accuracy:", test_accuracy)
        print("Test Precision:", test_precision)
        print("Test Recall:", test_recall)
        print("Test F1 Score:", test_f1)

        # Visualizar la cantidad de vectores de soporte
        plt.figure(figsize=(10, 5))

        # Gráfica de precisión
        plt.subplot(1, 2, 1)
        plt.plot(svm.n_support_, label='Number of Support Vectors')
        plt.xlabel('Class')
        plt.ylabel('Number of Support Vectors')
        plt.title('Number of Support Vectors')

        # Gráfica de pérdida
        plt.subplot(1, 2, 2)
        plt.plot(svm.support_, svm.dual_coef_.ravel(), 'o')
        plt.xlabel('Support Vector Index')
        plt.ylabel('Dual Coefficient')
        plt.title('Support Vectors')

        plt.tight_layout()
        plt.show()

    def run(self):

        while self.pronostico_activo:
            # Obtener nuevos datos para el pronóstico
            X_pronostico, Y_pronostico = self.obtener_datos_nuevos()  # Función para obtener los datos nuevos
            if len(X_pronostico)>=1:
                # Realizar el pronóstico

                resultado_pronostico = self.realizar_pronosticoCNN(self.modelo_entrenado, X_pronostico)
                print(resultado_pronostico)
                print(f"Resultado pronostico: {np.round(resultado_pronostico)}")
                presicion = np.round(resultado_pronostico)
                print(f"presición: {np.mean(presicion == resultado_pronostico)}")
                # Obtener la clase con la mayor probabilidad
                #clase_predicha = np.argmax(resultado_pronostico, axis=1)

                # Obtener la probabilidad más alta
                #certeza = np.max(resultado_pronostico, axis=1)

                # Imprimir resultados
                # for pronostico in resultado_pronostico:
                #     print("Pronóstico:", pronostico)
                #     print(f"Clase predicha: {clase_predicha}\nCerteza: {certeza} ")
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