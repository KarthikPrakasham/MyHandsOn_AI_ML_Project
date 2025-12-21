import streamlit as st
import pandas as pd
import joblib
st.title("JanataHack Prediction App")
dataFrame = pd.read_csv("resources/train.csv")

city = st.selectbox('City',dataFrame['city'].unique())
city_development_index = st.number_input('City Development Index')
gender = st.selectbox('Gender',dataFrame['gender'].unique())
relevent_experience = st.selectbox('Relevent Experience',dataFrame['relevent_experience'].unique())
enrolled_university = st.selectbox('Enrolled University',dataFrame['enrolled_university'].unique())
education_level = st.selectbox('Education Level',dataFrame['education_level'].unique())
major_discipline = st.selectbox('Major Discipline',dataFrame['major_discipline'].unique())
experience = st.selectbox('Experience',dataFrame['experience'].unique())
company_size = st.selectbox('Company Size',dataFrame['company_size'].unique())
company_type = st.selectbox('Company Type',dataFrame['company_type'].unique())
last_new_job = st.selectbox('Last New Job',dataFrame['last_new_job'].unique())
training_hours = st.number_input('Training Hours')

inputs = {
    'city': city,
    'city_development_index': city_development_index,
    'gender': gender,
    'relevent_experience': relevent_experience,
    'enrolled_university': enrolled_university,
    'education_level': education_level,
    'major_discipline': major_discipline,
    'experience': experience,
    'company_size': company_size,
    'company_type': company_type,
    'last_new_job': last_new_job,
    'training_hours': training_hours
}

if st.button('Predict'):
    input_df = pd.DataFrame([inputs])
    model = joblib.load('janatahack_model.pkl')
    prediction = model.predict(input_df)
    st.write(f'Prediction: {prediction[0]}')
