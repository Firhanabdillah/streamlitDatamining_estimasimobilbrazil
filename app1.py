
import pandas as pd
import numpy as np
import pickle


import streamlit as st


from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


from PIL import Image

# ----------------------------------------------------------------------------------


path = 'carros.csv'
df = pd.read_csv(path, encoding='iso-8859-1')


image = Image.open('mobil.jpg')

st.image(image)

# ----------------------------------------------------------------------------------
# Texto inicial
# ----------------------------------------------------------------------------------
"""
##  | Selamat Datang 
Silakan isi kebutuhan mobil yang anda inginkan lalu tekan button "Kirim".


"""



# ----------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------
st.title('HARGA MOBIL ANDA INGINKAN ')
"""

"""


filename = 'RF_car_prices1.1.sav'
model = pickle.load(open(filename, 'rb'))

with st.form("my_form"):
    
    i_motor = st.sidebar.select_slider("Tenaga motor",
                                options= sorted( df.motor.unique() ))

    
    i_cambio = st.sidebar.radio('TIPE MOBIL',
                                ('Manual', 'Automátis'))
    if i_cambio == 'Manual':
        i_cambio = 0
    else:
        i_cambio = 1


    i_ano = st.sidebar.select_slider("Tahun pembuatan",
                                     options= sorted( df.ano_fabrica.unique() ),
                                     value=2020 )

    i_km = st.sidebar.text_input('Kilometer', '10550')

    i_fabricante = st.sidebar.selectbox('Pabrikan', sorted( df.fabricante.unique() ) )

    i_cat = st.sidebar.selectbox('Model', sorted( df.car[df.fabricante == i_fabricante].unique() ) )
    i_cat = df.categoria[df.car == i_cat].values[0]


    i_combustivel = st.sidebar.selectbox('Jenis bahan bakar', df.combustivel.unique() )

    i_anunciante = st.sidebar.radio('Pilih Pembelian :',
                                    ('Individu', 'Perusahaan', 'Deealer') )


    i_regiao = st.sidebar.radio('Di wilayah Brasil manakah Anda berada?',
                                ('Utara', 'Timur Laut', 'Barat Tengah', 'Tenggara', 'Selatan') )

    df_to_predict = pd.DataFrame({'motor': [i_motor], 'automatico': [i_cambio], 'ano_fabrica': [i_ano], 'km': [int(i_km)],
                                  'categoria_Camionete':[1 if i_cat == 'Camionete' else 0], 'categoria_Carro':[1 if i_cat == 'Carro' else 0],
                                  'categoria_Minivan':[1 if i_cat == 'Minivan' else 0], 'categoria_SUV':[1 if i_cat == 'SUV' else 0],
                                  'fabricante_AUDI': [1 if i_fabricante == 'AUDI' else 0],'fabricante_BMW': [1 if i_fabricante == 'BMW' else 0],
                                  'fabricante_CHERY': [1 if i_fabricante == 'CHERY' else 0], 'fabricante_CHEVROLET': [1 if i_fabricante == 'CHEVROLET' else 0],
                                  'fabricante_CITROEN': [1 if i_fabricante == 'CITROEN' else 0],'fabricante_DODGE': [1 if i_fabricante == 'DODGE' else 0],
                                  'fabricante_FIAT': [1 if i_fabricante == 'FIAT' else 0], 'fabricante_FORD': [1 if i_fabricante == 'FORD' else 0],
                                  'fabricante_HONDA': [1 if i_fabricante == 'HONDA' else 0], 'fabricante_HYUNDAI': [1 if i_fabricante == 'HYUNDAI' else 0],
                                  'fabricante_JAC': [1 if i_fabricante == 'JAC' else 0], 'fabricante_JAGUAR': [1 if i_fabricante == 'JAGUAR' else 0],
                                  'fabricante_JEEP': [1 if i_fabricante == 'JEEP' else 0], 'fabricante_KIA': [1 if i_fabricante == 'KIA' else 0],
                                  'fabricante_LAND ROVER': [1 if i_fabricante == 'LAND ROVER' else 0], 'fabricante_LEXUS': [1 if i_fabricante == 'LEXUS' else 0],
                                  'fabricante_LIFAN': [1 if i_fabricante == 'LIFAN' else 0], 'fabricante_MERCEDES-BENZ': [1 if i_fabricante == 'MERCEDES-BENZ' else 0],
                                  'fabricante_MINI': [1 if i_fabricante == 'MINI' else 0],'fabricante_MITSUBISHI': [1 if i_fabricante == 'MITSUBISHI' else 0],
                                  'fabricante_NISSAN': [1 if i_fabricante == 'NISSAN' else 0], 'fabricante_PEUGEOT': [1 if i_fabricante == 'PEUGEOT' else 0],
                                  'fabricante_PORSCHE': [1 if i_fabricante == 'PORSCHE' else 0], 'fabricante_RENAULT': [1 if i_fabricante == 'RENAULT' else 0],
                                  'fabricante_SUZUKI': [1 if i_fabricante == 'SUZUKI' else 0], 'fabricante_TOYOTA': [1 if i_fabricante == 'TOYOTA' else 0],
                                  'fabricante_VOLKSWAGEN': [1 if i_fabricante == 'VOLKSWAGEN' else 0],'fabricante_VOLVO': [1 if i_fabricante == 'VOLVO' else 0],
                                  'combustivel_DIESEL': [1 if i_combustivel == 'DIESEL' else 0], 'combustivel_FLEX': [1 if i_combustivel == 'FLEX' else 0],
                                  'combustivel_GASOLINA': [1 if i_combustivel == 'GASOLINA' else 0], 'combustivel_HIBRIDO': [1 if i_combustivel == 'HIBRIDO' else 0],
                                  'anunciante_Concessionária': [1 if i_anunciante == 'Concessionária' else 0], 'anunciante_Loja': [1 if i_anunciante == 'Loja' else 0],'anunciante_Pessoa Física': [1 if i_anunciante == 'Pessoa Física' else 0],
                                  'regiao_Centro-Oeste': [1 if i_regiao == 'Centro-Oeste' else 0], 'regiao_Nordeste': [1 if i_regiao == 'Nordeste' else 0],
                                  'regiao_Norte': [1 if i_regiao == 'Norte' else 0], 'regiao_Sudeste': [1 if i_regiao == 'Sudeste' else 0], 'regiao_Sul': [1 if i_regiao == 'Sul' else 0]})


with st.form(key='from_prediction'):
    X = pd.get_dummies(df_to_predict)
    previsao = model.predict(X)
    previsao = previsao.tolist()[0]
    previsao_rupiah = round(previsao*14500)
    previsao_rupiah_str = f' Rp {previsao_rupiah:,}'
    previsao_dollar = round(previsao)
    previsao_dollar_str = f' R$ {previsao_dollar:,}'
    submitted = st.form_submit_button("KIRIM")
    if submitted:
        st.title(previsao_rupiah_str)
        st.title(previsao_dollar_str)








