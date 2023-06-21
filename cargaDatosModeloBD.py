import math
import os
import pandas as pd
from sqlalchemy import create_engine
import db_connection
import pandas as pn
from pandas.io import sql
def insertaRegistrosModelo():
    cnx = create_engine("postgresql+psycopg2://etvmoabf:etvmoabf@localhost/etvmoabf")
    datosCSV = "Final_datos_entrenamiento_v3.csv"
    try:
        df = pd.read_csv(datosCSV, sep=',')
        #print(df.to_string())
        #print(cnx)
        registros_Insertados = df.to_sql(con=cnx, name='datos_pronostico', schema='dbsds', if_exists='append', index=False)
        print(f"se insertaron {registros_Insertados} registros.")
    except ValueError:
        print(f"Error: {ValueError}\nejecutando el archivo (File): {datosCSV}")




def calcular_distancia(latitud1, longitud1, latitud2, longitud2):
    # Radio de la Tierra en kilómetros
    radio_tierra = 6371

    # Convertir las coordenadas a radianes
    latitud1 = math.radians(latitud1)
    longitud1 = math.radians(longitud1)
    latitud2 = math.radians(latitud2)
    longitud2 = math.radians(longitud2)

    # Diferencias entre las latitudes y longitudes
    delta_latitud = latitud2 - latitud1
    delta_longitud = longitud2 - longitud1

    # Fórmula de Haversine
    a = math.sin(delta_latitud/2)**2 + math.cos(latitud1) * math.cos(latitud2) * math.sin(delta_longitud/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    # Distancia entre los dos puntos
    distancia = radio_tierra * c

    return distancia
def cargar_csv_y_calcular_distancia(ruta_csv, nombreArchivo):
    print('Iniciando calculo de distancias')
    # Cargar el archivo CSV en un DataFrame de Pandas
    df = pd.read_csv(os.path.join(ruta_csv, nombreArchivo))

    # Calcular la distancia entre los puntos y asignarla a la columna 'dpr_distanciaentreobj'
    df['dpr_distanciaentreobj'] = df.apply(lambda row: calcular_distancia(row['dpr_latitudobja'], row['dpr_longitudobja'], row['dpr_latitudobjb'], row['dpr_longitudobjb']), axis=1)

    # Guardar el DataFrame actualizado en un nuevo archivo CSV
    df.to_csv(os.path.join(ruta_csv, 'Final_'+nombreArchivo), index=False)
    print('calculo de distancias finalizado.')

    return df

#cargar_csv_y_calcular_distancia('', 'datos_entrenamiento_v3.csv')
insertaRegistrosModelo()