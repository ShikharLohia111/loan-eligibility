import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import load_model
import tensorflow as tf
import pickle

# Load the pre-trained Keras model
model = tf.keras.models.load_model('loan_eligibility_model.keras')

# Load label encoder (you need to save the LabelEncoder used for encoding categorical variables)

# Function to encode the categorical columns
def encode_input(data):
    # Encode categorical variables
    print(data)
    cols_to_encode = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    for col in cols_to_encode:
        with open(f'label_encoder_${col}.pkl', 'rb') as file:
            le = pickle.load(file)
        data[col] = le.transform(data[col])
    return data

# Function to make predictions
def predict_eligibility(input_data):
    # Preprocess input data the same way as during model training
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    input_data_scaled = scaler.transform(input_data)
    st.write(input_data)
    # Get prediction from the model (output 0 or 1)
    prediction = model.predict(input_data_scaled)
    st.write(prediction)
    if prediction >= 0.5:
        return "Eligible for loan"
    else:
        return "Not eligible for loan"


# Streamlit app UI
st.title("Loan Eligibility Prediction")

# Input fields for the user
gender = st.selectbox("Gender", ('Male', "Female"))
married = st.selectbox("Marital Status", ("Yes", "No"))
dependents = st.selectbox("Dependents", ("0", "1", "2", "3+"))
education = st.selectbox("Education", ("Graduate", "Not Graduate"))
self_employed = st.selectbox("Self Employed", ("Yes", "No"))
property_area = st.selectbox("Property Area", ("Urban", "Semiurban", "Rural"))
loan_amount = st.number_input("Loan Amount", min_value=0, step=1000)
applicant_income=st.number_input("Applicant Income", min_value=0, step=100)
coapplicant_income=st.number_input("Coapplicant Income", min_value=0, step=100)
loan_term = st.number_input("Loan Term (in months)", min_value=0, step=1)
credit_history = st.selectbox("Credit History", ("1", "0"))

# Create a DataFrame from user input
user_input = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area],
})

# Encode categorical variables
print(user_input)
user_input = encode_input(user_input)
print(user_input)
# Make prediction when the button is clicked
if st.button('Predict Eligibility'):
    # Preprocess the input (you may need to scale the loan amount or other numeric columns like you did for training)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    user_input_scaled = scaler.fit_transform(user_input[['LoanAmount', 'Loan_Amount_Term']])

    # Get prediction and display result
    eligibility = predict_eligibility(user_input)
    st.write(f"Prediction: {eligibility}")
