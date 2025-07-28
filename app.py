import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
pipeline = joblib.load('patient_visit_model.joblib')

# Load the list of patient names
with open('patient_names.pkl', 'rb') as f:
    patient_names = pickle.load(f)

st.title('Patient Visit Prediction App')

st.write("Enter the details to predict patient visit count.")

# Add input widgets for user
# We will ask for Patient Name and Date to derive the temporal features

# Patient Name selection
selected_patient = st.selectbox('Select Patient Name', patient_names)

# Date input
selected_date = st.date_input('Select Date')

# Create a button to trigger prediction
if st.button('Predict Visit Count'):
    # Prepare the input data for prediction
    # We need to create a DataFrame with the same structure as the training data features
    input_data = pd.DataFrame({
        'PATIENT NAME': [selected_patient],
        'year': [selected_date.year],
        'month': [selected_date.month],
        'day_of_week': [selected_date.weekday()], # Monday=0, Sunday=6
        'week_of_year': [selected_date.isocalendar().week],
        'quarter': [selected_date.quarter]
    })

    # Ensure 'week_of_year' is integer type
    input_data['week_of_year'] = input_data['week_of_year'].astype(int)


    # Make prediction
    predicted_visit_count = pipeline.predict(input_data)

    # Display the prediction
    # Since visit count is an integer, let's display it as such, rounding the prediction
    st.subheader('Predicted Visit Count:')
    st.write(f"The predicted visit count for {selected_patient} on {selected_date.strftime('%Y-%m-%d')} is: {max(0, round(predicted_visit_count[0]))}") # Ensure prediction is not negative
