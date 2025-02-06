import streamlit as st
import pandas as pd
import numpy as np
import pickle  # Assuming the model is saved as a pickle file

# Load the trained ML model
with open("flight_delay_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

st.title("Flight Delay Prediction")

# Input fields
airline = st.selectbox("Select Airline", ["Airline A", "Airline B", "Airline C"])  # Add actual airline names
origin = st.text_input("Enter Origin Airport Code")
destination = st.text_input("Enter Destination Airport Code")
departure_date = st.date_input("Select Departure Date")
departure_time = st.time_input("Select Departure Time")
flight_duration = st.number_input("Enter Flight Duration (in minutes)", min_value=0)

# Predict button
if st.button("Predict Delay"):
    # Convert input into model features (Modify as per your model's feature engineering)
    input_data = pd.DataFrame({
        "Airline": [airline],
        "Origin": [origin],
        "Destination": [destination],
        "Departure_Date": [departure_date.strftime('%Y-%m-%d')],
        "Departure_Time": [departure_time.strftime('%H:%M:%S')],
        "Flight_Duration": [flight_duration],
    })

    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    # Display result
    if prediction[0] == 1:
        st.error(f"The flight is predicted to be delayed with a probability of {probability[0][1] * 100:.2f}%")
    else:
        st.success(f"The flight is predicted to be on time with a probability of {probability[0][0] * 100:.2f}%")
