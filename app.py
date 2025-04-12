import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load model and feature columns
@st.cache_resource
def load_model():
    model = joblib.load('heart_disease_rf_model.pkl')
    with open('feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
    return model, feature_columns

model, feature_columns = load_model()

# App title
st.title('Heart Disease Risk Prediction')
st.markdown('Predict the likelihood of heart disease based on your medical metrics.')

# Sidebar Inputs
st.sidebar.header('Enter Patient Details')

age = st.sidebar.slider('Age', 20, 100, 50)
resting_bp = st.sidebar.slider('Resting BP (mm Hg)', 90, 200, 120)
cholesterol = st.sidebar.slider('Cholesterol (mg/dl)', 100, 600, 200)
max_hr = st.sidebar.slider('Max Heart Rate Achieved', 60, 220, 150)
oldpeak = st.sidebar.slider('ST Depression (Oldpeak)', 0.0, 6.2, 1.0)

sex = st.sidebar.selectbox('Sex', ['Female', 'Male'])
fasting_bs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl?', ['No', 'Yes'])
exercise_angina = st.sidebar.selectbox('Exercise-Induced Angina?', ['No', 'Yes'])

chest_pain_type = st.sidebar.selectbox('Chest Pain Type', [
    'Typical Angina (0)', 'Atypical Angina (1)', 'Non-anginal Pain (2)', 'Asymptomatic (3)'])

resting_ecg = st.sidebar.selectbox('Resting ECG', [
    'Normal (0)', 'ST-T Abnormality (1)', 'LV Hypertrophy (2)'])

st_slope = st.sidebar.selectbox('ST Slope', [
    'Upsloping (0)', 'Flat (1)', 'Downsloping (2)'])

# Encode categorical inputs
sex = 1 if sex == 'Male' else 0
fasting_bs = 1 if fasting_bs == 'Yes' else 0
exercise_angina = 1 if exercise_angina == 'Yes' else 0
chest_pain_type = int(chest_pain_type.split('(')[1][0])
resting_ecg = int(resting_ecg.split('(')[1][0])
st_slope = int(st_slope.split('(')[1][0])

# Feature Engineering
rate_pressure_product = (resting_bp * max_hr) / 100
heart_rate_reserve = 220 - age - max_hr

# One-hot encode
chest_pain_dummies = [0, 0, 0]
chest_pain_dummies[chest_pain_type] = 1

resting_ecg_dummies = [0, 0, 0]
resting_ecg_dummies[resting_ecg] = 1

st_slope_dummies = [0,0, 0, 0]
st_slope_dummies[st_slope] = 1

# Create a dictionary for input values
input_dict = dict(zip([
    'age',
    'resting_bp',
    'cholesterol',
    'max_hr',
    'oldpeak',
    'sex',
    'fasting_bs',
    'exercise_angina',
    'chest_pain_type_0',
    'chest_pain_type_1',
    'chest_pain_type_2',
    'resting_ecg_0',
    'resting_ecg_1',
    'resting_ecg_2',
    'st_slope_-1',
    'st_slope_0',
    'st_slope_1',
    'st_slope_2',
    'rate_pressure_product',
    'heart_rate_reserve'
], [
    age,
    resting_bp,
    cholesterol,
    max_hr,
    oldpeak,
    sex,
    fasting_bs,
    exercise_angina,
    *chest_pain_dummies,
    *resting_ecg_dummies,
    *st_slope_dummies,
    rate_pressure_product,
    heart_rate_reserve
]))

# Fill missing columns with 0
for col in feature_columns:
    if col not in input_dict:
        input_dict[col] = 0

# Reorder by feature_columns
input_df = pd.DataFrame([input_dict])[feature_columns]
input_df = input_df.astype(int)
# Predict
if st.sidebar.button('Predict'):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader('Prediction Result')
    if prediction == 1:
        st.error('⚠️ High Risk of Heart Disease Detected!')
        st.write(f'**Probability:** {proba * 100:.2f}%')
    else:
        st.success('✅ Low Risk of Heart Disease Detected.')
        st.write(f'**Probability:** {proba * 100:.2f}%')

    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        st.subheader("Top Influencing Features")
        importances = model.named_steps['classifier'].feature_importances_
        top_features = np.argsort(importances)[::-1][:5]
        for i in top_features:
            st.write(f"**{feature_columns[i]}:** {importances[i]:.4f}")

# Footer
st.markdown("""
---
**Note:** This tool provides a data-driven prediction and is not a substitute for clinical diagnosis.
""")
