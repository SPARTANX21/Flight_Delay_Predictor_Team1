import streamlit as st
import pandas as pd
import pickle
import base64

# Function to encode the image in base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Set the path to your image
image_path = r"C:\Users\Hites\Downloads\wallpaperflare.com_wallpaper.jpg"

# Get the base64 string
base64_img = get_base64_image(image_path)

# Apply the background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_img}"); 
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained ML model
with open("flight_delay_pipeline.pkl", "rb") as model_file:
    model = pickle.load(model_file)

st.title("Flight Delay Prediction")

# Sample list of airports
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

# Input fields with blank options before selection
airline = st.selectbox("Select Airline", ["", "Airline A", "Airline B", "Airline C"])
origin = st.selectbox("Select Origin Airport", [""] + airports)
destination = st.selectbox("Select Destination Airport", [""] + airports)
departure_date = st.date_input("Select Departure Date", value=None)

# Using columns to place time and AM/PM in one line
col1, col2 = st.columns([3, 1])

with col1:
    departure_time = st.time_input("Select Departure Time", value=None)

with col2:
    am_pm = st.selectbox("AM/PM", ["AM", "PM"])

# Combine time and AM/PM to create a 12-hour formatted time string
if departure_time:
    hour = departure_time.hour
    minute = departure_time.minute
    if am_pm == "PM" and hour < 12:
        hour += 12  # Adjust for PM times
    elif am_pm == "AM" and hour == 12:
        hour = 0  # Adjust for midnight (12 AM)
    departure_time_formatted = f"{hour:02d}:{minute:02d}"
else:
    departure_time_formatted = ""

flight_duration = st.number_input("Enter Flight Duration (in minutes)", min_value=0)

# Predict button
if st.button("Predict Delay"):
    if not origin or not destination:
        st.error("Please select both origin and destination airport.")
    elif flight_duration <= 0:
        st.error("Please enter a valid flight duration.")
    else:
        input_data = pd.DataFrame({
            "Airline": [airline],
            "Origin": [origin],
            "Destination": [destination],
            "Departure_Date": [departure_date.strftime('%Y-%m-%d')] if departure_date else "",
            "Departure_Time": departure_time_formatted,
            "Flight_Duration": [flight_duration],
        })

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.error(f"The flight is predicted to be delayed with a probability of {probability[0][1] * 100:.2f}%")
        else:
            st.success(f"The flight is predicted to be on time with a probability of {probability[0][0] * 100:.2f}%")
