import pandas as pd
from sqlalchemy import create_engine
import db_connection
import pandas as pn
from pandas.io import sql
def insertaRegistrosModelo():
    cnx = create_engine("postgresql+psycopg2://etvmoabf:etvmoabf@localhost/etvmoabf")
    datosCSV = "datos_entrenamiento_v2.csv"
    try:
        df = pd.read_csv(datosCSV, sep=',')
        print(df.to_string())
        print(cnx)
        registros_Insertados = df.to_sql(con=cnx, name='datos_pronostico', schema='dbsds', if_exists='append', index=False)
        print(f"se insertaron {registros_Insertados} registros.")
    except ValueError:
        print(f"Error: {ValueError}\nejecutando el archivo (File): {datosCSV}")

insertaRegistrosModelo()