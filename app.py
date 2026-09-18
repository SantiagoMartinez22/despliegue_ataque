# Despliegue del modelo de predicción de ataque al corazón
# - Cargamos el modelo
# - Capturamos los datos futuros con Streamlit
# - Preparamos los datos: dummies + reindex a las variables del entrenamiento
# - Aplicamos el modelo para la predicción

import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# 1. Cargamos el modelo
# ----------------------------------------------------------------------
st.set_page_config(page_title="Predicción de ataque al corazón", page_icon="❤️")


@st.cache_resource
def cargar_modelo():
    filename = "modelo-class.pkl"
    modelo, label_encoder, variables, min_max_scaler = pickle.load(open(filename, "rb"))
    return modelo, label_encoder, variables, min_max_scaler


modelo, label_encoder, variables, min_max_scaler = cargar_modelo()

# ----------------------------------------------------------------------
# 2. Interfaz gráfica para la captura de los datos
# ----------------------------------------------------------------------
st.title("Predicción de ataque al corazón")
st.write("Ingrese los datos del paciente y presione **Predecir**.")

age = st.slider("Edad (age)", min_value=1, max_value=82, value=45, step=1)

avg_glucose_level = st.slider(
    "Nivel promedio de glucosa (avg_glucose_level)",
    min_value=55.0,
    max_value=272.0,
    value=100.0,
    step=0.01,
)

hypertension = st.selectbox("Hipertensión (hypertension)", ["No", "Yes"])
heart_disease = st.selectbox("Enfermedad cardiaca (heart_disease)", ["No", "Yes"])
ever_married = st.selectbox("Alguna vez casado (ever_married)", ["No", "Yes"])
smoking_status = st.selectbox(
    "Hábito de fumar (smoking_status)",
    ["Unknown", "'never smoked'", "'formerly smoked'", "smokes"],
)

# Dataframe con los mismos nombres de variables del entrenamiento
datos = [[age, hypertension, heart_disease, ever_married, avg_glucose_level, smoking_status]]
data = pd.DataFrame(
    datos,
    columns=[
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "avg_glucose_level",
        "smoking_status",
    ],
)

st.subheader("Datos ingresados")
st.dataframe(data, use_container_width=True)

# ----------------------------------------------------------------------
# 3. Preparación de los datos
# ----------------------------------------------------------------------
data_preparada = data.copy()

# En despliegue drop_first = False
data_preparada = pd.get_dummies(
    data_preparada,
    columns=["hypertension", "heart_disease", "ever_married", "smoking_status"],
    drop_first=False,
    dtype=int,
)

# Se adicionan las columnas faltantes y se ordenan igual que en el entrenamiento
data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

# El modelo (XGBoost) se entrenó con las variables sin normalizar.
# Si se usara Knn, Red Neuronal, SVM o Regresión, se descomenta la línea siguiente:
# data_preparada[['age', 'avg_glucose_level']] = min_max_scaler.transform(
#     data_preparada[['age', 'avg_glucose_level']])

# ----------------------------------------------------------------------
# 4. Predicción
# ----------------------------------------------------------------------
if st.button("Predecir", type="primary"):
    Y_pred = modelo.predict(data_preparada)
    etiqueta = label_encoder.inverse_transform(Y_pred)[0]

    # Probabilidad de la clase positiva ('Yes')
    proba = modelo.predict_proba(data_preparada)[0]
    indice_yes = list(label_encoder.classes_).index("Yes")
    prob_yes = float(proba[indice_yes])

    data["Prediccion"] = etiqueta

    st.subheader("Resultado")
    if etiqueta == "Yes":
        st.error(f"Predicción: **SÍ** hay riesgo de ataque al corazón (probabilidad: {prob_yes:.2%})")
    else:
        st.success(f"Predicción: **NO** hay riesgo de ataque al corazón (probabilidad de riesgo: {prob_yes:.2%})")

    st.dataframe(data, use_container_width=True)

    # Recordar la medida de error del modelo
    st.warning(
        "Este resultado es una estimación estadística del modelo y no constituye "
        "un diagnóstico médico. Consulte siempre a un profesional de la salud."
    )

# ----------------------------------------------------------------------
# 5. Predicción masiva con un archivo (opcional)
# ----------------------------------------------------------------------
with st.expander("Predicción masiva (cargar archivo de datos futuros)"):
    archivo = st.file_uploader("Archivo .xlsx o .csv", type=["xlsx", "csv"])
    if archivo is not None:
        if archivo.name.endswith(".csv"):
            futuros = pd.read_csv(archivo)
        else:
            futuros = pd.read_excel(archivo)

        # Se elimina la variable objetivo si viene en el archivo
        futuros = futuros.drop(columns=["stroke_ataque_corazon"], errors="ignore")

        prep = pd.get_dummies(
            futuros,
            columns=["hypertension", "heart_disease", "ever_married", "smoking_status"],
            drop_first=False,
            dtype=int,
        ).reindex(columns=variables, fill_value=0)

        pred = label_encoder.inverse_transform(modelo.predict(prep))
        futuros["Prediccion"] = pred

        st.dataframe(futuros, use_container_width=True)
        st.download_button(
            "Descargar predicciones",
            futuros.to_csv(index=False).encode("utf-8"),
            file_name="predicciones_ataque_corazon.csv",
            mime="text/csv",
        )
