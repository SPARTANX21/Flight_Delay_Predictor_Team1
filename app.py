import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from xgboost import XGBClassifier

# Load taxi data at the beginning (after model loading)
taxi_df = pd.read_csv('Taxi_IN_OUT_MeanValues.csv')
# Load pre-trained models and encoders
preprocessor = joblib.load('preprocessor.joblib')
label_encoder = joblib.load('label_encoder.joblib')
model = joblib.load('xgboost_model.joblib')

# Airline and airport options
airlines = ['Alaska Airlines Inc.', 'Southwest Airlines Co.', 'United Air Lines Inc.',
            'Delta Air Lines Inc.', 'Republic Airlines', 'Endeavor Air Inc.',
            'Frontier Airlines Inc.', 'ExpressJet Airlines Inc.', 'JetBlue Airways',
            'SkyWest Airlines Inc.', 'Hawaiian Airlines Inc.', 'Horizon Air',
            'Virgin America', 'Allegiant Air', 'Spirit Air Lines',
            'GoJet Airlines, LLC d/b/a United Express', 'Trans States Airlines',
            'Mesa Airlines Inc.', 'Compass Airlines', 'Air Wisconsin Airlines Corp',
            'Commutair Aka Champlain Enterprises, Inc.', 'Peninsula Airways Inc.',
            'Cape Air', 'American Airlines Inc.', 'Envoy Air', 'Comair Inc.',
            'Capital Cargo International', 'Empire Airlines Inc.']

airports = ['ABE', 'ABI', 'ABQ', 'ABR', 'ABY', 'ACK', 'ACT', 'ACV', 'ACY', 'ADK', 'ADQ', 'AEX', 'AGS', 'AKN', 'ALB',
            'ALO', 'ALW', 'AMA', 'ANC', 'APN', 'ART', 'ASE', 'ATL', 'ATW', 'AUS', 'AVL', 'AVP', 'AZA', 'AZO', 'BDL',
            'BET', 'BFF', 'BFL', 'BGM', 'BGR', 'BHM', 'BIL', 'BIS', 'BJI', 'BKG', 'BLI', 'BLV', 'BMI', 'BNA', 'BOI',
            'BOS', 'BPT', 'BQK', 'BQN', 'BRD', 'BRO', 'BRW', 'BTM', 'BTR', 'BTV', 'BUF', 'BUR', 'BWI', 'BZN', 'CAE',
            'CAK', 'CDC', 'CDV', 'CGI', 'CHA', 'CHO', 'CHS', 'CID', 'CIU', 'CKB', 'CLE', 'CLL', 'CLT', 'CMH', 'CMI',
            'CMX', 'CNY', 'COD', 'COS', 'COU', 'CPR', 'CRP', 'CRW', 'CSG', 'CVG', 'CWA', 'CYS', 'DAB', 'DAL', 'DAY',
            'DBQ', 'DCA', 'DEN', 'DFW', 'DHN', 'DIK', 'DLG', 'DLH', 'DRO', 'DRT', 'DSM', 'DTW', 'DUT', 'DVL', 'EAR',
            'EAT', 'EAU', 'ECP', 'EGE', 'EKO', 'ELM', 'ELP', 'ERI', 'ESC', 'EUG', 'EVV', 'EWN', 'EWR', 'EYW', 'FAI',
            'FAR', 'FAT', 'FAY', 'FCA', 'FLG', 'FLL', 'FLO', 'FNT', 'FSD', 'FSM', 'FWA', 'GCC', 'GCK', 'GEG', 'GFK',
            'GGG', 'GJT', 'GNV', 'GPT', 'GRB', 'GRI', 'GRK', 'GRR', 'GSO', 'GSP', 'GST', 'GTF', 'GTR', 'GUC', 'GUM',
            'HDN', 'HGR', 'HHH', 'HIB', 'HLN', 'HNL', 'HOB', 'HOU', 'HPN', 'HRL', 'HSV', 'HTS', 'HVN', 'HYA', 'HYS',
            'IAD', 'IAG', 'IAH', 'ICT', 'IDA', 'ILM', 'IMT', 'IND', 'INL', 'IPT', 'ISN', 'ISP', 'ITH', 'ITO', 'JAC',
            'JAN', 'JAX', 'JFK', 'JHM', 'JLN', 'JMS', 'JNU', 'KOA', 'KTN', 'LAN', 'LAR', 'LAS', 'LAW', 'LAX', 'LBB',
            'LBE', 'LBF', 'LBL', 'LCH', 'LCK', 'LEX', 'LFT', 'LGA', 'LGB', 'LIH', 'LIT', 'LNK', 'LNY', 'LRD', 'LSE',
            'LWB', 'LWS', 'LYH', 'MAF', 'MBS', 'MCI', 'MCO', 'MDT', 'MDW', 'MEI', 'MEM', 'MFE', 'MFR', 'MGM', 'MHK',
            'MHT', 'MIA', 'MKE', 'MKG', 'MKK', 'MLB', 'MLI', 'MLU', 'MMH', 'MOB', 'MOT', 'MQT', 'MRY', 'MSN', 'MSO',
            'MSP', 'MSY', 'MTJ', 'MVY', 'MYR', 'OAJ', 'OAK', 'OGD', 'OGG', 'OGS', 'OKC', 'OMA', 'OME', 'ONT', 'ORD',
            'ORF', 'ORH', 'OTH', 'OTZ', 'OWB', 'PAH', 'PBG', 'PBI', 'PDX', 'PGD', 'PGV', 'PHF', 'PHL', 'PHX', 'PIA',
            'PIB', 'PIE', 'PIH', 'PIT', 'PLN', 'PNS', 'PPG', 'PQI', 'PRC', 'PSC', 'PSE', 'PSG', 'PSM', 'PSP', 'PUB',
            'PUW', 'PVD', 'PVU', 'PWM', 'RAP', 'RDD', 'RDM', 'RDU', 'RFD', 'RHI', 'RIC', 'RKS', 'RNO', 'ROA', 'ROC',
            'ROP', 'ROW', 'RST', 'RSW', 'SAF', 'SAN', 'SAT', 'SAV', 'SBA', 'SBN', 'SBP', 'SBY', 'SCC', 'SCE', 'SCK',
            'SDF', 'SEA', 'SFB', 'SFO', 'SGF', 'SGU', 'SHD', 'SHV', 'SIT', 'SJC', 'SJT', 'SJU', 'SLC', 'SLN', 'SMF',
            'SMX', 'SNA', 'SPI', 'SPN', 'SPS', 'SRQ', 'STC', 'STL', 'STS', 'STT', 'STX', 'SUN', 'SUX', 'SWF', 'SWO',
            'SYR', 'TLH', 'TOL', 'TPA', 'TRI', 'TTN', 'TUL', 'TUS', 'TVC', 'TWF', 'TXK', 'TYR', 'TYS', 'UIN', 'USA',
            'VEL', 'VLD', 'VPS', 'WRG', 'WYS', 'XNA', 'YAK', 'YKM', 'YNG', 'YUM']


def parse_time(time_str):
    """Convert HH:MM format to total minutes and HHMM string"""
    if not time_str:
        return 0, '0000'
    try:
        hours, minutes = map(int, time_str.split(':'))
        total_minutes = hours * 60 + minutes
        return total_minutes, f"{hours:02d}{minutes:02d}"
    except:
        return 0, '0000'


def calculate_wheels_off_on(schd_dep_time, schd_arr_time, taxi_out=10, taxi_in=20):
    dep_mins, dep_hhmm = parse_time(schd_dep_time)
    arr_mins, arr_hhmm = parse_time(schd_arr_time)

    # WheelsOff = SchdDepTime + TaxiOut
    wheels_off_mins = dep_mins + taxi_out
    wheels_off_hh = (wheels_off_mins // 60) % 24
    wheels_off_mm = wheels_off_mins % 60
    wheels_off = f"{wheels_off_hh:02d}{wheels_off_mm:02d}"

    # WheelsOn calculation
    flight_duration = arr_mins - dep_mins
    actual_flight_duration = flight_duration - (taxi_out + taxi_in)
    wheels_on_mins = wheels_off_mins + actual_flight_duration
    wheels_on_hh = (wheels_on_mins // 60) % 24
    wheels_on_mm = wheels_on_mins % 60
    wheels_on = f"{wheels_on_hh:02d}{wheels_on_mm:02d}"

    return wheels_off, wheels_on


def get_schd_dep_time_of_day(hour):
    if 0 <= hour < 6:
        return 'Night'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'


# Streamlit UI
st.title('Flight Delay Prediction')

# Input fields
airline = st.selectbox('Airline', airlines)
origin = st.selectbox('Origin Airport', airports)
dest = st.selectbox('Destination Airport', airports)
date = st.date_input('Travel Date', datetime.today())
schd_dep_time = st.text_input('Scheduled Departure Time (HH:MM)', '12:00')
schd_arr_time = st.text_input('Scheduled Arrival Time (HH:MM)', '14:00')


def validate_inputs(origin, dest, schd_dep_time, schd_arr_time, travel_date):
    errors = []

    # Check if origin and destination are the same
    if origin == dest:
        errors.append("Origin and Destination airports cannot be the same.")

    # Check if scheduled departure and arrival times are the same
    if schd_dep_time >= schd_arr_time:
        errors.append("Scheduled Departure Time and Scheduled Arrival Time cannot be the same.")

    # Check if travel date is in the past
    current_date = datetime.now().date()
    if travel_date < current_date:
        errors.append("Travel Date must be today or a future date.")

    return errors

# Modify the prediction button section
if st.button('Predict Delay Probability'):
    # Validate inputs
    validation_errors = validate_inputs(origin, dest, schd_dep_time, schd_arr_time, date)

    if validation_errors:
        for error in validation_errors:
            st.error(error)
    else:
        # Date processing
        year = date.year
        quarter = (date.month - 1) // 3 + 1
        month = date.month
        dayofmonth = date.day
        dayofweek = date.isoweekday()  # Monday=1, Sunday=7

        # Time processing
        schd_dep_hour = int(schd_dep_time.split(':')[0]) if ':' in schd_dep_time else 0
        schd_dep_time_of_day = get_schd_dep_time_of_day(schd_dep_hour)

        # Get TaxiIn and TaxiOut from CSV
        try:
            # Look for matching Origin-Dest pair
            taxi_data = taxi_df[(taxi_df['Origin'] == origin) & (taxi_df['Dest'] == dest)]

            if not taxi_data.empty:
                # Take first matching row's values
                taxi_in = round(taxi_data.iloc[0]['TaxiIn'], 2)
                taxi_out = round(taxi_data.iloc[0]['TaxiOut'], 2)
            else:
                # Default values if no match found
                taxi_in = 10
                taxi_out = 15

        except Exception as e:
            st.error(f"Error loading taxi data: {str(e)}")
            taxi_in = 10
            taxi_out = 15

        # Calculate derived features
        wheels_off, wheels_on = calculate_wheels_off_on(schd_dep_time, schd_arr_time, taxi_out, taxi_in)
        is_weekend = 1 if dayofweek in [6, 7] else 0

        # Create input DataFrame
        input_data = pd.DataFrame([{
            'Airline': airline,
            'Origin': origin,
            'Dest': dest,
            'Year': year,
            'Quarter': quarter,
            'Month': month,
            'DayofMonth': dayofmonth,
            'DayOfWeek': dayofweek,
            'TaxiOut': taxi_out,
            'WheelsOff': int(wheels_off),
            'WheelsOn': int(wheels_on),
            'TaxiIn': taxi_in,
            'SchdArrTime': int(schd_arr_time.replace(':', '')),
            'SchdDepHour': schd_dep_hour,
            'IsWeekend': is_weekend,
            'SchdDepTimeOfDay': schd_dep_time_of_day
        }])

        # Preprocess and predict
        processed_data = preprocessor.transform(input_data)
        probabilities = model.predict_proba(processed_data)[0]

        # Create result display with correct ordering
        result_labels = [
            'No Delay(0-15min)',
            'Short Delay(15-60min)',
            'Medium Delay(60-120min)',
            'Long Delay(120-180min)',
            'Very Long Delay(180min and more)'
        ]

        # Map probabilities to correct order based on label encoder
        label_order = label_encoder.classes_
        prob_mapping = {
            'No Delay(0-15min)': probabilities[np.where(label_order == 'No Delay')[0][0]],
            'Short Delay(15-60min)': probabilities[np.where(label_order == 'Short Delay')[0][0]],
            'Medium Delay(60-120min)': probabilities[np.where(label_order == 'Medium Delay')[0][0]],
            'Long Delay(120-180min)': probabilities[np.where(label_order == 'Long Delay')[0][0]],
            'Very Long Delay(180min and more)': probabilities[np.where(label_order == 'Very Long Delay')[0][0]]
        }

        # Display results
        st.subheader('Arrival Delay Chances:')
        for label, prob in prob_mapping.items():
            st.write(f"{label} = {prob * 100:.2f}%")