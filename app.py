# Importar biblioteca completa
import streamlit as st
import pandas as pd

# Importar algo especifico de uma biblioteca
from sklearn.model_selection import train_test_split

# Criar Funções (df) 
def data_split (df, teste_size):
    SEED = 1561651
    treino_df, teste_df = train_test_split(df, teste_size = teste_size, random_state=SEED)
    return treino_df.reset_index(drop=True), teste_df.reset_index(drop=True)

def pipeline_teste(df):

    pipeline = Pipeline([
        ('feature_dropper', DropFeatures()),
        ('OneHotEncoding', OneHotEncodingNames()),
        ('ordinal_feature', OrdinalFeature()),
        ('min_max_scaler', MinMaxWithFeatNames()),
    ])
    df_pipeline = pipeline.fit_transform(df)
    return df_pipeline


# Importar base tratada
link = 'https://raw.githubusercontent.com/RicardViana/fiap-data-viz-and-production-models/refs/heads/main/df_clean.csv'
dados = pd.read_csv(link, sep= ",")

# Criar a aplicação streamlit
st.write('# Simulador de avaliação de crédito')

# Criar perguntas idade
st.write('### Idade')
input_idade = float(st.slider('Selecione sua idade',18,100))

# Criar pergunta escolaridade
st.write('### Nível de escolaridade')
input_grau_escolaridade = st.selectbox('Qual é o seu grau de escolaridade', dados['Grau_escolaridade'].unique())

# Criar pergunta estado civiel
st.write('### Estado civil')
input_estado_civil = st.selectbox('Qual é o seu estado civil', dados['Estado_civil'].unique())

# Criar pergunta sobre familia
st.write('### Família')
input_membros_familia = float(st.slider('Selecione quantos membros tem na sua família',1,20))

# Criar pergunta sobre carro proprio
st.write('### Carro próprio')
input_carro_proprio_dict = {'Sim': 1, 'Não': 0}
input_carro_proprio = st.radio('Você possui um automóvel?', ['Sim', 'Não'])
input_carro_proprio = input_carro_proprio_dict.get(input_carro_proprio)

# Criar pergunta sobre casa
st.write('### Casa própria')
input_casa_propria_dict = {'Sim': 1, 'Não': 0}
input_casa_propria = st.radio('Você possui um propriedade?', ['Sim', 'Não'])
input_casa_propria = input_casa_propria_dict.get(input_casa_propria)

# Criar pergunta tipo de residencia
st.write('### Tipo de residência')
input_tipo_moradia = st.selectbox('Qual é o seu tipo de moradia?', dados['Moradia'].unique())

# Criar pergunta tipo de categoria de renda
st.write('### Tipo de categoria de renda')
input_categoria_renda = st.selectbox('Qual é o categoria de renda?', dados['Categoria_de_renda'].unique())

# Criar pergunta tipo de ocupação
st.write('### Tipo de ocupação')
input_ocupacao = st.selectbox('Qual é o seu tipo de ocupação?', dados['Ocupacao'].unique())

# Criar pergunta tempo de experiência
st.write('### Experiência')
input_tempo_experiencia = float(st.slider('Qual é o seu tempo de experiência?', 0 , 30))

# Criar pergunta sobre rendimento
st.write('### Rendimento')
input_rendimentos = float(st.number_input('Digite o seu rendimento anual (em reais) e pressione ENTER para confirmar', 0))

# Criar pergunta telefone corporativo
st.write('### Telefone corporativo')
input_telefone_trabalho_dict = {'Sim': 1, 'Não': 0}
input_telefone_trabalho = st.radio('Você possui um telefone corporativo?', ['Sim', 'Não'])
input_telefone_trabalho = input_telefone_trabalho_dict.get(input_telefone_trabalho)

# Criar pergunta telefone fixo
st.write('### Telefone fixo')
input_telefone_dict = {'Sim': 1, 'Não': 0}
input_telefone = st.radio('Você possui um telefone fixo?', ['Sim', 'Não'])
input_telefone = input_telefone_dict.get(input_telefone)

# Criar pergunta e-mail
st.write('### Email')
input_email_dict = {'Sim': 1, 'Não': 0}
input_email = st.radio('Você tem um Email?', ['Sim', 'Não'])
input_email = input_email_dict.get(input_email)

# Criar lista na ordem do data frame
novo_cliente = [
    0,
    input_carro_proprio,
    input_casa_propria,
    input_telefone_trabalho,
    input_telefone,
    input_email,
    input_membros_familia,
    input_rendimentos,
    input_idade,
    input_tempo_experiencia,
    input_categoria_renda,
    input_grau_escolaridade,
    input_estado_civil,
    input_tipo_moradia,
    input_ocupacao,
    0
]

# Separar a base em treino e teste
treino_df, teste_df = data_split(dados, 0.2)
cliente_predict_df = pd.DataFrame([novo_cliente], columns=teste_df.colums)
teste_novo_cliente = pd.concat([teste_df, cliente_predict_df], columns=teste_df.columns)

# Rodar o pipeline
teste_novo_cliente = pipeline_teste(teste_novo_cliente)
cliente_pred = teste_novo_cliente.drop(['Mau'], axis =1)

# Fazer a predição
if st.button('Enviar'):
    model = joblib.load('modelo/xgb.joblib')
    final_pred = model.predict(cliente_pred)
    if final_pred[-1] == 0:
        st.success('### Parabéns! Você teve o cartão de crédito aprovado')
        st.balloons()

