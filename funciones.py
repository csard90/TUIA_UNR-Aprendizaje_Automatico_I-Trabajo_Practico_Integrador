import pandas as pd
import joblib
import warnings
warnings.simplefilter('ignore')

# Basicamente aca estan las mismas codificaciones que estaban repartidas a lo largo del notebook pero 
# todas juntas (o casi todas). No nos alcanzo el tiempo como para hacer las clases y el pipeline.

# Categorizar direcciones del viento:
def categorizar_direcciones(direccion):

    if direccion in ['N', 'NNE', 'NE', 'NNW', 'NW']:
        return 'Norte'
    elif direccion in ['S', 'SSE', 'SE', 'SSW', 'SW']:
        return 'Sur'
    elif direccion in ['E', 'ENE', 'ESE']:
        return 'Este'
    elif direccion in ['W', 'WNW', 'WSW']:
        return 'Oeste'
    else:
        return 'Otras'

# Funcion para categorizar fecha segun estacion:
def asignar_estacion(fecha):

    if fecha.month == 12 and fecha.day >= 21 or fecha.month == 1 or fecha.month == 2 or (fecha.month == 3 and fecha.day < 21):
        return 'Verano'
    elif fecha.month == 3 and fecha.day >= 21 or fecha.month == 4 or fecha.month == 5 or (fecha.month == 6 and fecha.day < 21):
        return 'Otoño'
    elif fecha.month == 6 and fecha.day >= 21 or fecha.month == 7 or fecha.month == 8 or (fecha.month == 9 and fecha.day < 21):
        return 'Invierno'
    elif fecha.month == 9 and fecha.day >= 21 or fecha.month == 10 or fecha.month == 11 or (fecha.month == 12 and fecha.day < 21):
        return 'Primavera'

def normalizacion_codificacion(dataframe):

    # ----------------------------------------------------------------------------------------------
    # Codificar RainToday:
    dataframe['RainToday'] = dataframe['RainToday'].apply(lambda x: 1 if x == 'Yes' else 0)
    # ----------------------------------------------------------------------------------------------


    # ----------------------------------------------------------------------------------------------
    # Codificacion de variables WindDir:
    dataframe['WindGustDir_Codificada'] = dataframe['WindGustDir'].apply(categorizar_direcciones)
    dataframe['WindDir3pm_Codificada'] = dataframe['WindDir3pm'].apply(categorizar_direcciones)
    dataframe['WindDir9am_Codificada'] = dataframe['WindDir9am'].apply(categorizar_direcciones)     
    # ----------------------------------------------------------------------------------------------
    # Crear dummies de variables WindDir:
    # ----------------------------------------------------------------------------------------------
    dummies_windgustdir = pd.get_dummies(dataframe['WindGustDir_Codificada'], prefix='WindGustDir')
    dataframe = pd.concat([dataframe, dummies_windgustdir], axis=1)

    dummies_winddir3pm = pd.get_dummies(dataframe['WindDir3pm_Codificada'], prefix='WindDir3pm')
    dataframe = pd.concat([dataframe, dummies_winddir3pm], axis=1)

    dummies_winddir9am = pd.get_dummies(dataframe['WindDir9am_Codificada'], prefix='WindDir9am')
    dataframe = pd.concat([dataframe, dummies_winddir9am], axis=1)
    # ----------------------------------------------------------------------------------------------


    # ----------------------------------------------------------------------------------------------
    # Categorizar estaciones:
    dataframe['Estacion'] = dataframe['Date'].apply(asignar_estacion)
    # ----------------------------------------------------------------------------------------------
    # Dummies de estaciones
    dummies_estacion = pd.get_dummies(dataframe['Estacion'], prefix='Estacion')
    dataframe = pd.concat([dataframe, dummies_estacion], axis=1)
    # ----------------------------------------------------------------------------------------------


    # ----------------------------------------------------------------------------------------------
    # Agregamos el resto de dummies que no se crearon:
    resto_columnas = ['WindGustDir_Este', 'WindGustDir_Norte', 'WindGustDir_Oeste',
                    'WindDir3pm_Este', 'WindDir3pm_Norte', 'WindDir3pm_Oeste',
                    'WindDir9am_Este', 'WindDir9am_Norte', 'WindDir9am_Oeste',
                    'Estacion_Invierno', 'Estacion_Otoño', 'Estacion_Primavera']
    
    for columna in resto_columnas:
        if columna not in dataframe.columns:
            dataframe[columna] = False
    # ----------------------------------------------------------------------------------------------


    
    # ----------------------------------------------------------------------------------------------
    # Variables con las que me voy a quedar luego de todo el proceso:
    filtro_caracteristicas = ['MinTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed',
                            'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                            'Pressure9am', 'Cloud9am', 'Cloud3pm', 'Temp3pm', 'RainToday',
                            'WindGustDir_Este', 'WindGustDir_Norte', 'WindGustDir_Oeste',
                            'WindDir3pm_Este', 'WindDir3pm_Norte', 'WindDir3pm_Oeste',
                            'WindDir9am_Este', 'WindDir9am_Norte', 'WindDir9am_Oeste',
                            'Estacion_Invierno', 'Estacion_Otoño', 'Estacion_Primavera']

    dataframe = dataframe[filtro_caracteristicas]
    # ----------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------
    # Importamos el escalador entrenado con los datos de entrenamiento:
    escalador = joblib.load('D:\\Users\\csard 90\\Desktop\\TUIA\\Materias\\IV\\IA 4.1 Apendizaje Automatico I\\TP\\escalador_entrenado.joblib')

    # Aunque no usemos esta variable para predecir la necesitamos porque el escalador fue entrenado con ella
    # (despues la borramos)
    dataframe['RainfallTomorrow'] = [0]

    # Escalamos los datos:
    columnas_para_estandarizar = ['MinTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Cloud9am', 'Cloud3pm', 'Temp3pm', 'RainToday', 'RainfallTomorrow']
    
    resto_de_columnas = ['WindGustDir_Este', 'WindGustDir_Norte', 'WindGustDir_Oeste',
        'WindDir3pm_Este', 'WindDir3pm_Norte',
        'WindDir3pm_Oeste', 'WindDir9am_Este',
        'WindDir9am_Norte', 'WindDir9am_Oeste',
        'Estacion_Invierno', 'Estacion_Otoño', 'Estacion_Primavera']
    
    # Guardamos en un df los datos escalados
    dataframe_z = pd.DataFrame(escalador.transform(dataframe[columnas_para_estandarizar]), columns=columnas_para_estandarizar)
    dataframe_z = pd.concat([dataframe_z.drop('RainfallTomorrow', axis=1), dataframe[resto_de_columnas].reset_index(drop=True)], axis=1)
    # ----------------------------------------------------------------------------------------------

    # Casteo al tipo de dato correcto:    
    dataframe_z[resto_de_columnas] = dataframe_z[resto_de_columnas].astype('uint8')
    # ----------------------------------------------------------------------------------------------
    
    return dataframe_z