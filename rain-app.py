import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('C:\\Users\\Shumakov_OA\\Desktop\\weatherAUS.csv', sep=',')

df = df.drop(['Date', 'Location'], axis=1)
df = df.dropna()
df = df.reset_index(drop=True)

wd = df['WindGustDir'].unique() 

st.write("""# Простое приложение для предсказания вероятности осадков на завтрашний день""")
st.sidebar.header('Состояние погоды на сегодняшний день')

def user_input_features():
    MinTemp = st.sidebar.slider('Минимальная тепература, С', float(df["MinTemp"].min()), float(df["MinTemp"].max()))
    MaxTemp = st.sidebar.slider('Максимальная температура, С', float(df["MaxTemp"].min()), float(df["MaxTemp"].max()))
    Rainfall = st.sidebar.slider('Осадки за сутки, мм', float(df["Rainfall"].min()), float(df["Rainfall"].max()))
    Evaporation = st.sidebar.slider('Испарение класса А за сутки, мм', float(df["Evaporation"].min()), float(df["Evaporation"].max()))
    Sunshine = st.sidebar.slider('Солнечное сияние, ч', float(df["Sunshine"].min()), float(df["Sunshine"].max()))
    WindGustDir = st.sidebar.radio('Направление сильнейшего порыва ветра', wd)
    WindGustSpeed = st.sidebar.slider('Скорость сильнейшего порыва ветра, км/ч', float(df["WindGustSpeed"].min()), float(df["WindGustSpeed"].max()))
    WindDir9am = st.sidebar.radio('Направление ветра в 9:00', wd)
    WindDir3pm = st.sidebar.radio('Направление ветра в 15:00', wd)
    WindSpeed9am = st.sidebar.slider('Скорость ветра в 9:00, км/ч', float(df["WindSpeed9am"].min()), float(df["WindSpeed9am"].max()))
    WindSpeed3pm = st.sidebar.slider('Скорость ветра в 15:00, км/ч', float(df["WindSpeed3pm"].min()), float(df["WindSpeed3pm"].max()))
    Humidity9am = st.sidebar.slider('Влажность воздуха в 9:00, %', float(df["Humidity9am"].min()), float(df["Humidity9am"].max()))
    Humidity3pm = st.sidebar.slider('Влажность воздуха в 15:00, %', float(df["Humidity3pm"].min()), float(df["Humidity3pm"].max()))
    Pressure9am = st.sidebar.slider('Атмосферное давление в 9:00, гПа', float(df["Pressure9am"].min()), float(df["Pressure9am"].max()))
    Pressure3pm = st.sidebar.slider('Атмосферное давление в 15:00, гПа', float(df["Pressure3pm"].min()), float(df["Pressure3pm"].max()))
    Cloud9am = st.sidebar.slider('Облачность в 9:00, октантов', float(df["Cloud9am"].min()), float(df["Cloud9am"].max()))
    Cloud3pm = st.sidebar.slider('Облачность в 15:00, октантов', float(df["Cloud3pm"].min()), float(df["Cloud3pm"].max()))
    Temp9am = st.sidebar.slider('Температура воздуха в 9:00, С', float(df["Temp9am"].min()), float(df["Temp9am"].max()))
    Temp3pm = st.sidebar.slider('Температура воздуха в 15:00, С', float(df["Temp3pm"].min()), float(df["Temp3pm"].max()))
    RainToday = st.sidebar.radio('Был ли дождь сегодня?', ['Yes', 'No'])
    data = {'MinTemp': MinTemp,
            'MaxTemp': MaxTemp,
            'Rainfall': Rainfall,
            'Evaporation': Evaporation,
            'Sunshine': Sunshine,
            'WindGustDir': WindGustDir,
            'WindGustSpeed': WindGustSpeed,
            'WindDir9am': WindDir9am,
            'WindDir3pm': WindDir3pm,
            'WindSpeed9am': WindSpeed9am,
            'WindSpeed3pm': WindSpeed3pm,
            'Humidity9am': Humidity9am,
            'Humidity3pm': Humidity3pm,
            'Pressure9am': Pressure9am,
            'Pressure3pm': Pressure3pm,
            'Cloud9am': Cloud9am,
            'Cloud3pm': Cloud3pm,
            'Temp9am': Temp9am,
            'Temp3pm': Temp3pm,
            'RainToday': RainToday}
    features = pd.DataFrame(data, index=['one'])
    return features

input_features = user_input_features()

st.subheader('Состояние погоды на сегодняшний день')
st.write(input_features)

df_ = pd.get_dummies(df, columns=["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"], prefix=["WG", "WD9am", "WD3pm", "Rain"])

X = df_.drop('RainTomorrow', 1)
Y = df_['RainTomorrow']

clf = RandomForestClassifier()
clf.fit(X, Y)

df = df.drop('RainTomorrow', 1)
df = pd.concat([df, input_features])
df = pd.get_dummies(df, columns=["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"], prefix=["WG", "WD9am", "WD3pm", "Rain"])
input_features = df.query("index == 'one'")

prediction = clf.predict(input_features)
prediction_proba = clf.predict_proba(input_features)
st.subheader('Прогноз осадков на завтрашний день')
st.write(prediction)
st.subheader('Вероятность осадков на завтрашний день')
st.write(prediction_proba)