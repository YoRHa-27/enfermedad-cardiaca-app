import streamlit as st
import joblib
import pandas as pd
import os

# Título de la aplicación
st.title('Predicción de Enfermedad Cardíaca')
st.write('Ingrese los datos del paciente para predecir la probabilidad de enfermedad cardíaca.')
st.write("**Nombre:** Sebastian Adrian Pastor Daza")
st.write("**Código ISIL:** 72481264")
st.markdown("**Cuaderno de Código:** https://colab.research.google.com/drive/1rgM0_jQf0xs7zZeEkIebsnCQgtFiC7KF?usp=sharing")

# --- Cargar Modelos ---
# Asegúrate de que la carpeta 'modelos' esté en el mismo directorio que este script o especifica la ruta completa.
model_path_lr = 'modelos/logistic_regression_model.pkl'
model_path_rf = 'modelos/random_forest_model.pkl'

# Verificar si los modelos existen
if not os.path.exists(model_path_lr) or not os.path.exists(model_path_rf):
    st.error("Error: No se encontraron los archivos de los modelos. Asegúrese de que la carpeta 'modelos' y los archivos .pkl estén en la ubicación correcta.")
    st.stop()

# Cargar los modelos
logistic_regression_model = joblib.load(model_path_lr)
random_forest_model = joblib.load(model_path_rf)

# --- Selector de Modelo ---
selected_model_name = st.selectbox(
    'Seleccione el modelo a utilizar:',
    ('Regresión Logística', 'Random Forest')
)

if selected_model_name == 'Regresión Logística':
    model = logistic_regression_model
else:
    model = random_forest_model

# --- Entradas del Usuario ---
st.sidebar.header('Datos del Paciente')

def user_input_features():
    age = st.sidebar.slider('Edad', 29, 77, 54)
    sex = st.sidebar.selectbox('Sexo (0=Mujer, 1=Hombre)', (0, 1))
    cp = st.sidebar.selectbox('Tipo de dolor de pecho (1=Angina Típica, 2=Angina Atípica, 3=Dolor No Anginoso, 4=Asintomático)', (1, 2, 3, 4))
    trestbps = st.sidebar.slider('Presión arterial en reposo (mm Hg)', 94, 200, 130)
    chol = st.sidebar.slider('Colesterol sérico (mg/dl)', 126, 564, 240)
    fbs = st.sidebar.selectbox('Azúcar en sangre en ayunas > 120 mg/dl (0=Falso, 1=Verdadero)', (0, 1))
    restecg = st.sidebar.selectbox('Resultados electrocardiográficos en reposo (0=Normal, 1=Anormalidad de la onda ST-T, 2=Hipertrofia ventricular izquierda)', (0, 1, 2))
    thalach = st.sidebar.slider('Frecuencia cardíaca máxima alcanzada', 71, 202, 150)
    exang = st.sidebar.selectbox('Angina inducida por ejercicio (0=No, 1=Sí)', (0, 1))
    oldpeak = st.sidebar.slider('Depresión del ST inducida por el ejercicio en relación con el reposo', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Pendiente del segmento ST del pico del ejercicio (1=Ascendente, 2=Plana, 3=Descendente)', (1, 2, 3))
    ca = st.sidebar.selectbox('Número de vasos principales coloreados por fluoroscopia (0, 1, 2, 3)', (0, 1, 2, 3))
    thal = st.sidebar.selectbox('Talasemia (3=Normal, 6=Defecto Fijo, 7=Defecto Reversible)', (3, 6, 7))

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('Datos de Entrada del Usuario')
st.write(input_df)

# --- Botón de Predicción ---
if st.button('Obtener Predicción'):
    prediction = model.predict(input_df)
    # El objetivo 'num' tiene valores de 0 a 4
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Resultado de la Predicción')
    st.write(f"El modelo predice una severidad de enfermedad cardíaca de: **{prediction[0]}**")
    st.write("Gravedad: | 0=Sin enfermedad | 1=Leve | 2=Moderada | 3=Grave | 4=Muy grave")
    st.write("Probabilidades para cada clase:")
    st.write(pd.DataFrame(prediction_proba, columns=['Clase 0', 'Clase 1', 'Clase 2', 'Clase 3', 'Clase 4'], index=['Probabilidad']))
