import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('models/trained_model.pkl')
st.title("California Housing Price Prediction")

# User inputs
latitude = st.slider('Latitude', 32.0, 42.0, 36.5)
longitude = st.slider('Longitude', -124.0, -114.0, -119.5)
housing_median_age = st.slider('Median Age of House', 1, 52, 20)
total_rooms = st.slider('Total Rooms', 1, 39320, 6000)
total_bkts = st.slider('Total Bkts', 1, 2330, 500)
population = st.slider('Population', 3, 35682, 1500)
households = st.slider('Households', 1, 6062, 500)
med_inc = st.slider('Median Income (in $10,000s)', 0.5, 15.0, 3.0)
ave_bedrms = st.slider('Average Bedrooms', 1.0, 10.0, 2.0)

input_dict = {
    'MedInc': med_inc,
    'HouseAge': housing_median_age,
    'AveRooms': total_rooms / total_bkts,
    'AveBedrms': ave_bedrms,
    'Population': population,
    'AveOccup': population / households,
    'Latitude': latitude,
    'Longitude': longitude
}


input_df = pd.DataFrame([input_dict])

# Predict
prediction = model.predict(input_df)
st.write(f"Estimated Median House Value: ${prediction[0]*100000:.2f}")