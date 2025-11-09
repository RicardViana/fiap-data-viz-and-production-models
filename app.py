import streamlit as st
import pandas as pd

# Importar base tratada
link = 'https://raw.githubusercontent.com/RicardViana/fiap-data-viz-and-production-models/refs/heads/main/df_clean.csv'
dados = pd.read_csv(link, sep= ",")

# Criar a aplicação streamlit
st.write('# Simulador de avaliação de crédito')

# Criar perguntas 
