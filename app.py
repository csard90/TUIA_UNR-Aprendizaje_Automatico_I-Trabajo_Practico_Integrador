import streamlit as st
import numpy as np
import joblib
import pandas as pd
from funciones import normalizacion_codificacion
import keras 
import warnings
warnings.simplefilter('ignore')

# pip install joblib
# pip install pandas
# pip install streamlit
# pip install tensorFlow

# --------------------------------------------------------------------------
# Carga de modelos:
regresion_lineal = joblib.load('D:\\Users\\csard 90\\Desktop\\TUIA\\Materias\\IV\\IA 4.1 Apendizaje Automatico I\\TP\\regresion_lineal.joblib')
regresion_logistica = joblib.load('D:\\Users\\csard 90\\Desktop\\TUIA\\Materias\\IV\\IA 4.1 Apendizaje Automatico I\\TP\\regresion_logistica.joblib')
# Para los modelos de redes neuronales se utilizo otra libreria porque joblib daba demasiados problemas.
# De la notebook tal como esta se descarga una carpeta comprimida, hay que descomprimirla y cargar la carpeta entera.
red_neuronal_clasificacion = keras.models.load_model('D:\\Users\\csard 90\\Desktop\\TUIA\\Materias\\IV\\IA 4.1 Apendizaje Automatico I\\TP\\red_neuronal_clasificacion')
red_neuronal_regresion = keras.models.load_model('D:\\Users\\csard 90\\Desktop\\TUIA\\Materias\\IV\\IA 4.1 Apendizaje Automatico I\\TP\\red_neuronal_regresion')
escalador = joblib.load('D:\\Users\\csard 90\\Desktop\\TUIA\\Materias\\IV\\IA 4.1 Apendizaje Automatico I\\TP\\escalador_entrenado.joblib')
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Titulo
st.title('Prediccion de lluvias en Australia')
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Opciones de direcciones del viento:
opciones_wind_gust_dir = ['S', 'NE', 'NNE', 'SSW', 'SSE', 'ENE', 'N', 'E', 'SE', 'WSW',
       'WNW', 'ESE', 'NNW', 'SW', 'NW', 'W']
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Ingreso de valores:
st.write('Ingresar valores sin estandarizar: ')
fecha = st.date_input(label='Date', format='YYYY-MM-DD')
min_temp = st.number_input(label='MinTemp')
rainfall = st.number_input(label='Rainfall', min_value=0.0)
evaporation = st.number_input(label='Evaporation', min_value=0.0)
sunshine = st.number_input(label='Sunshine', min_value=0.0)
wind_gust_speed = st.number_input(label='WindGustSpeed', min_value=0.0)
wind_speed_9am = st.number_input(label='WindGustSpeed9am', min_value=0.0)
wind_speed_3pm = st.number_input(label='WindGustSpeed3pm', min_value=0.0)
humidity_9am = st.number_input(label='Humidity9am', min_value=0.0)
humidity_3pm = st.number_input(label='Humidity3pm', min_value=0.0)
pressure_9am = st.number_input(label='Pressure9am', min_value=0.0)
cloud_9am = st.number_input(label='Cloud9am', min_value=0.0)
cloud_3pm = st.number_input(label='Cloud3pm', min_value=0.0)
temp_3pm = st.number_input(label='Temp3pm', min_value=0.0)
rain_today = st.selectbox(label='Raintoday', options=['Yes', 'No'])
wind_gust_dir = st.selectbox(label='WindGustDir', options=opciones_wind_gust_dir)
wind_dir_9am = st.selectbox(label='WindGustDir9am', options=opciones_wind_gust_dir)
wind_dir_3pm = st.selectbox(label='WindGustDir3pm', options=opciones_wind_gust_dir)
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Si se confirma que los datos ingresados son los que queremos verificar:
if st.button('Consultar prediccion'):

    # --------------------------------------------------------------------------
    # Guardamos todos los valores en una lista:
    fila = [fecha, min_temp, rainfall, evaporation, sunshine, wind_gust_speed, wind_speed_9am, 
            wind_speed_3pm, humidity_9am, humidity_3pm, pressure_9am, cloud_9am, cloud_3pm, temp_3pm, rain_today,
            wind_gust_dir, wind_dir_9am, wind_dir_3pm]
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # Dataframe para guardar los datos
    datos = pd.DataFrame( columns= [ 'Date', 'MinTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 
                                'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',  
                                'Cloud9am', 'Cloud3pm', 'Temp3pm', 'RainToday', 'WindGustDir', 'WindDir9am', 
                                'WindDir3pm' ] )
    # --------------------------------------------------------------------------

    # Ingresar una fila en el dataset en el ultimo indice para poder hacer la codificacion
    datos.loc[len(datos)] = fila

    # Codificacion:
    datos = normalizacion_codificacion(datos)
    
    # --------------------------------------------------------------------------
    # Predicciones de la regresion lineal (recordar que viene estandarizada)
    prediccion_rl_z = regresion_lineal.predict(datos)[0]
    
    # Convertimos a la escala original con el escalador que traiamos del notebook. 
    # Recordar que el escalador funcionaba con 15 caracteristicas y la numero 15 (posicion 14)
    # era la variable RainfallTomorrow, por lo tanto llenamos un array con cualquier valor, calculamos
    # y nos quedamos con la posicion que nos interesa.
    prediccion_rl = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, prediccion_rl_z])
    prediccion_rl = escalador.inverse_transform(prediccion_rl.reshape(1, -1))[0][14]

    # Mostramos por pantalla ambos valores
    st.write(f"Prediccion regresion lineal: escala z={round(prediccion_rl_z, 2)} | escala original={ round(prediccion_rl, 2)}")
    # --------------------------------------------------------------------------

    # Predicciones de la red neuronal de regresion:
    prediccion_rn_r_z = red_neuronal_regresion.predict(datos.values)[0][0]
    
    # Igual que arriba:
    prediccion_rn_r = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, prediccion_rn_r_z])
    prediccion_rn_r = escalador.inverse_transform(prediccion_rn_r.reshape(1, -1))[0][14]

    st.write(f"Prediccion rn regresion: escala z={round(prediccion_rn_r_z, 2)} | escala original={round(prediccion_rn_r, 2)}")
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Prediccion regresion logistica
    prediccion_rlog = regresion_logistica.predict(datos)[0]
    

    # Convertimos el valor a categoria
    if prediccion_rlog == 1: 
        prediccion_rlog = 'Llueve' 
    else:
        prediccion_rlog = 'No llueve'

    st.write(f'Prediccion regresion logistica: {prediccion_rlog}')

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Prediccion red neuronal clasificacion:
    prediccion_rn_c = red_neuronal_clasificacion.predict(datos.values)[0][0]

    prediccion_rn_c = (prediccion_rn_c >= 0.5).astype(int)    
    if prediccion_rn_c == 1: 
        prediccion_rn_c = 'Llueve' 
    else:
        prediccion_rn_c = 'No llueve'

    st.write(f'Prediccion rn clasificacion: {prediccion_rn_c}')

    # --------------------------------------------------------------------------


