import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
# ==========================
# CONFIGURAÇÃO DA PÁGINA
# ==========================

st.set_page_config(
    page_title="Predição de Obesidade",
    layout="wide"
)

# ==========================
# LEITURA DOS DADOS
# ==========================

df = pd.read_csv("Obesity.csv")

df["BMI"] = df["Weight"] / (df["Height"] ** 2)
modelo = joblib.load("modelo_obesidade.pkl")

# ==========================
# MENU LATERAL
# ==========================

pagina = st.sidebar.radio(
    "Menu",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🤖 Predição"
    ]
)

# ==========================
# HOME
# ==========================

if pagina == "🏠 Home":

    st.title("Predição de Obesidade")

    st.markdown("---")

    st.header("Tech Challenge - FIAP Pós Tech Data Analytics")

    st.write("""
    Este projeto tem como objetivo auxiliar profissionais da saúde
    na identificação do nível de obesidade de pacientes utilizando
    técnicas de Machine Learning.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Pacientes Analisados", "2.087")

    with col2:
        st.metric("Acurácia do Modelo", "98%")

    with col3:
        st.metric("Classes Previstas", "7")

# ==========================
# DASHBOARD
# ==========================

elif pagina == "📊 Dashboard":

    st.title("Dashboard Analítico")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "IMC Médio",
            round(df["BMI"].mean(), 2)
        )

    with col2:
        st.metric(
            "Peso Médio",
            round(df["Weight"].mean(), 2)
        )

    with col3:
        st.metric(
            "Idade Média",
            round(df["Age"].mean(), 2)
        )

    st.markdown("---")

    fig1 = px.histogram(
        df,
        x="Obesity",
        title="Distribuição dos Níveis de Obesidade"
    )

    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.box(
        df,
        x="Obesity",
        y="BMI",
        title="IMC por Nível de Obesidade"
    )

    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(
        df,
        x="family_history",
        color="Obesity",
        barmode="group",
        title="Histórico Familiar x Obesidade"
    )

    st.plotly_chart(fig3, use_container_width=True)

# ==========================
# PREDIÇÃO
# ==========================

elif pagina == "🤖 Predição":

    st.title("Sistema Preditivo")

    st.markdown("""
    Preencha as informações do paciente para realizar a avaliação.
    """)

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gênero",
            ["Male", "Female"]
        )

        age = st.number_input(
            "Idade",
            min_value=14,
            max_value=80,
            value=25
        )

        height = st.number_input(
            "Altura (m)",
            min_value=1.40,
            max_value=2.10,
            value=1.70
        )

        weight = st.number_input(
            "Peso (kg)",
            min_value=30.0,
            max_value=250.0,
            value=80.0
        )

        family_history = st.selectbox(
            "Histórico Familiar",
            ["yes", "no"]
        )

    with col2:

        favc = st.selectbox(
            "Consome alimentos calóricos frequentemente?",
            ["yes", "no"]
        )

        ch2o = st.slider(
            "Consumo de Água",
            1.0,
            3.0,
            2.0
        )

        faf = st.slider(
            "Atividade Física",
            0.0,
            3.0,
            1.0
        )

        calc = st.selectbox(
            "Consumo de Álcool",
            ["no", "Sometimes", "Frequently", "Always"]
        )

    bmi = weight / (height ** 2)

    st.metric(
        "IMC Calculado",
        round(bmi, 2)
    )

    if st.button("Realizar Predição"):

        entrada = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'family_history': [family_history],
            'FAVC': [favc],
            'FCVC': [2.0],
            'NCP': [3.0],
            'CAEC': ['Sometimes'],
            'SMOKE': ['no'],
            'CH2O': [ch2o],
            'SCC': ['no'],
            'FAF': [faf],
            'TUE': [1.0],
            'CALC': [calc],
            'MTRANS': ['Public_Transportation'],
            'BMI': [bmi]
        })

        resultado = modelo.predict(entrada)

        st.success(
            f"Nível previsto: {resultado[0]}"
        )
