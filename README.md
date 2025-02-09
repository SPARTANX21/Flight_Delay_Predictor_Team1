
# Flight Delay Predictor

**Repository**: Machine Learning Project on Predicting Flight Delays

Flight delays and cancellations have been a persistent challenge in the aviation industry, causing widespread disruptions, financial losses, and reducing customer satisfaction. Understanding the factors that contribute to these delays is essential for improving operational efficiency and enhancing the travel experience.

This project aims to develop a machine learning model that predicts the probability of flight delays. By analyzing historical flight data, key variables such as airline performance, departure schedules, and airport traffic will be used to provide accurate predictions. The ultimate goal is to empower passengers to make informed travel decisions while helping airlines optimize their operations.

## Project Overview

The **Flight Delay Dataset (2018–2022)** provides a comprehensive repository of historical flight data, enabling an in-depth analysis of delay patterns and trends across the United States. The dataset includes a wide range of critical features such as:

- Airline performance
- Flight schedules
- Departure and arrival delays
- Airport congestion
- Seasonal variations

With over four years of detailed data, the project will explore these insights to develop a robust machine learning model. This model will help airlines identify potential bottlenecks and improve schedule reliability while enhancing customer satisfaction.

### Key Features of the Project:
- **Predictive Model**: Leveraging machine learning algorithms to predict flight delays and cancellations.
- **User Interface**: A user-friendly interface that empowers passengers with flight delay predictions and helps airlines make data-driven decisions.
- **Operational Insights**: Use analytics to identify key factors contributing to delays and cancellations, providing actionable insights for airlines.
- **Data Analysis**: Explore seasonal variations, airport congestion, and other key variables to improve operational resilience and enhance the passenger experience.

---

## Data Dictionary

This dataset provides a wealth of features that describe the flight operations and delays in detail. Here's a summary of the columns available in the dataset:

| **Column Name**                                      | **Description**                                                                                                                                                             |
|------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Year**                                             | Year of the flight.                                                                                                                                                          |
| **Quarter**                                          | Quarter of the year (1-4).                                                                                                                                                   |
| **Month**                                            | Month of the flight.                                                                                                                                                         |
| **DayofMonth**                                       | Day of the month.                                                                                                                                                           |
| **DayOfWeek**                                        | Day of the week.                                                                                                                                                            |
| **FlightDate**                                       | Flight date in the format `yyyymmdd`.                                                                                                                                           |
| **Marketing_Airline_Network**                        | Unique marketing carrier code (e.g., PA, PA(1), PA(2)) to distinguish different carriers.                                                                                     |
| **Operated_or_Branded_Code_Share_Partners**          | Reporting carrier operated or branded code share partners.                                                                                                                  |
| **DOT_ID_Marketing_Airline**                         | DOT-assigned ID number for a unique airline (carrier).                                                                                                                       |
| **IATA_Code_Marketing_Airline**                      | IATA-assigned code for identifying a carrier.                                                                                                                                |
| **Flight_Number_Marketing_Airline**                  | Flight number for the marketing airline.                                                                                                                                     |
| **Originally_Scheduled_Code_Share_Airline**          | Unique scheduled operating carrier code with possible numeric suffixes.                                                                                                     |
| **DOT_ID_Originally_Scheduled_Code_Share_Airline**   | DOT ID for the originally scheduled code-share airline.                                                                                                                     |
| **IATA_Code_Originally_Scheduled_Code_Share_Airline**| IATA code for the originally scheduled code-share airline.                                                                                                                  |
| **Flight_Num_Originally_Scheduled_Code_Share_Airline**| Flight number for the originally scheduled code-share airline.                                                                                                              |
| **Operating_Airline**                                | Unique carrier code for the operating airline, with possible numeric suffixes.                                                                                              |
| **DOT_ID_Operating_Airline**                         | DOT identification number for the operating airline.                                                                                                                        |
| **IATA_Code_Operating_Airline**                      | IATA code for the operating airline.                                                                                                                                         |
| **Tail_Number**                                      | Aircraft tail number.                                                                                                                                                       |
| **Flight_Number_Operating_Airline**                  | Flight number for the operating airline.                                                                                                                                     |
| **OriginAirportID**                                  | Unique ID for the origin airport, assigned by US DOT.                                                                                                                       |
| **OriginAirportSeqID**                               | Sequence ID for the origin airport, identifying time-specific details.                                                                                                      |
| **OriginCityMarketID**                               | City market ID for the origin airport, consolidating airports serving the same city market.                                                                                  |
| **Origin**                                           | Origin airport code.                                                                                                                                                        |
| **OriginCityName**                                   | Name of the city for the origin airport.                                                                                                                                     |
| **OriginState**                                      | State code for the origin airport.                                                                                                                                          |
| **OriginStateFips**                                  | State FIPS code for the origin airport.                                                                                                                                      |
| **OriginStateName**                                  | Full state name for the origin airport.                                                                                                                                      |
| **OriginWac**                                         | World area code (WAC) for the origin airport.                                                                                                                                |
| **DestAirportID**                                    | Unique ID for the destination airport, assigned by US DOT.                                                                                                                 |
| **DestAirportSeqID**                                 | Sequence ID for the destination airport, identifying time-specific details.                                                                                                 |
| **DestCityMarketID**                                 | City market ID for the destination airport, consolidating airports serving the same city market.                                                                            |
| **Dest**                                             | Destination airport code.                                                                                                                                                   |
| **DestCityName**                                     | Name of the city for the destination airport.                                                                                                                                |
| **DestState**                                        | State code for the destination airport.                                                                                                                                     |
| **DestStateFips**                                    | State FIPS code for the destination airport.                                                                                                                                |
| **DestStateName**                                    | Full state name for the destination airport.                                                                                                                                |
| **DestWac**                                          | World area code (WAC) for the destination airport.                                                                                                                           |
| **CRSDepTime**                                       | Scheduled departure time (local time, hhmm).                                                                                                                                 |
| **DepTime**                                          | Actual departure time (local time, hhmm).                                                                                                                                   |
| **DepDelay**                                         | Difference in minutes between scheduled and actual departure time. Negative values indicate early departures.                                                              |
| **DepDelayMinutes**                                  | Departure delay in minutes. Early departures are set to 0.                                                                                                                 |
| **DepDel15**                                         | Departure delay indicator (1 = delay of 15 minutes or more).                                                                                                                |
| **DepartureDelayGroups**                             | Departure delay intervals (in 15-minute increments up to 180).                                                                                                            |
| **DepTimeBlk**                                       | Scheduled departure time block (hourly intervals).                                                                                                                          |
| **TaxiOut**                                          | Taxi-out time in minutes.                                                                                                                                                   |
| **WheelsOff**                                        | Wheels-off time (local time, hhmm).                                                                                                                                          |
| **WheelsOn**                                         | Wheels-on time (local time, hhmm).                                                                                                                                           |
| **TaxiIn**                                          | Taxi-in time in minutes.                                                                                                                                                    |
| **CRSArrTime**                                       | Scheduled arrival time (local time, hhmm).                                                                                                                                   |
| **ArrTime**                                          | Actual arrival time (local time, hhmm).                                                                                                                                      |
| **ArrDelay**                                         | Difference in minutes between scheduled and actual arrival time. Negative values indicate early arrivals.                                                                  |
| **ArrDelayMinutes**                                  | Arrival delay in minutes. Early arrivals are set to 0.                                                                                                                     |
| **ArrDel15**                                         | Arrival delay indicator (1 = delay of 15 minutes or more).                                                                                                                 |
| **ArrivalDelayGroups**                               | Arrival delay intervals (in 15-minute increments up to 180).                                                                                                               |
| **ArrTimeBlk**                                       | Scheduled arrival time block (hourly intervals).                                                                                                                           |
| **Cancelled**                                        | Flight cancellation indicator (1 = Yes).                                                                                                                                     |
| **CancellationCode**                                 | Reason for flight cancellation.                                                                                                                                             |
| **Diverted**                                         | Diverted flight indicator (1 = Yes).                                                                                                                                         |
| **CRSElapsedTime**                                   | Scheduled elapsed flight time in minutes.                                                                                                                                   |
| **ActualElapsedTime**                                | Actual elapsed flight time in minutes.                                                                                                                                      |
| **AirTime**                                          | Actual flight time in minutes.                                                                                                                                             |
| **Flights**                                          | Number of flights.                                                                                                                                                         |
| **Distance**                                         | Distance between origin and destination airports (in miles).                                                                                                               |
| **DistanceGroup**                                    | Distance intervals (every 250 miles).                                                                                                                                       |                                                                                                                           |

---

## Key Features of the Model

- **Prediction Variables**: Analyze critical factors such as flight departure and arrival times, carrier delays, weather delays, and operational metrics.
- **Machine Learning Algorithms**: Use classification algorithms like Decision Trees, Random Forest, and Logistic Regression to predict delays or cancellations.
- **Data Exploration & Visualization**: Utilize data visualizations to understand delay trends, carrier performance, and seasonal variations.
- **User Interface**: Create a web-based or mobile interface where users can input flight details and get an estimated delay prediction.

---
