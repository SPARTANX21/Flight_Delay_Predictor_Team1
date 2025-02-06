import streamlit as st
import pandas as pd
import pickle

# Sample list of airports (Add more or modify as needed)
airports = [
    "Los Angeles International Airport (LAX)",
    "John F. Kennedy International Airport (JFK)",
    "San Francisco International Airport (SFO)",
    "Chicago O'Hare International Airport (ORD)",
    "Miami International Airport (MIA)",
    "Dallas/Fort Worth International Airport (DFW)",
    "Denver International Airport (DEN)",
    "Seattle-Tacoma International Airport (SEA)",
    "Boston Logan International Airport (BOS)",
    "Atlanta Hartsfield-Jackson International Airport (ATL)"
]

# Set background image using an absolute path
st.markdown(
    """
    <style>
    .stApp {
        background-image: url(https://www.google.com/search?sa=X&sca_esv=0af04e46daa0125b&udm=2&sxsrf=AHTn8zqnTDCYukudrY2NYElpwJ1bDDblVg:1738837523580&q=high+resolution+airplane+wallpaper&stick=H4sIAAAAAAAAAFvEqpSRmZ6hUJRanJ9TWpKZn6eQmFlUkJOYl6pQnpiTU5BYkFoEAOnrMSElAAAA&source=univ&ved=2ahUKEwjf66jH6q6LAxW4yzgGHdfKMtUQrNwCegQIFRAA&biw=1536&bih=730&dpr=1.25#vhid=dmWhVW6qnGsk8M&vssid=mosaic);
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained ML model
with open("flight_delay_pipeline.pkl", "rb") as model_file:
    model = pickle.load(model_file)

st.title("Flight Delay Prediction")

# Input fields
airline = st.selectbox("Select Airline", ["Airline A", "Airline B", "Airline C"])  # Add actual airline names

# Origin Airport Dropdown
origin = st.selectbox("Select Origin Airport", airports)

# Destination Airport Dropdown
destination = st.selectbox("Select Destination Airport", airports)

# Departure Date Calendar
departure_date = st.date_input("Select Departure Date")

# Departure Time Clock
departure_time = st.time_input("Select Departure Time")

# Flight Duration Input
flight_duration = st.number_input("Enter Flight Duration (in minutes)", min_value=0)

# Predict button
if st.button("Predict Delay"):
    # Validate inputs
    if not origin or not destination:
        st.error("Please select both origin and destination airport.")
    elif flight_duration <= 0:
        st.error("Please enter a valid flight duration.")
    else:
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
