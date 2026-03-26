import json
import pickle

import numpy as np
import streamlit as st

# Load model
with open("banglore_home_prices_model.pickle", "rb") as f:
    model = pickle.load(f)

# Load columns
with open("columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]

# Extract locations
locations = data_columns[3:]   # first 3 are sqft, bath, bhk

st.set_page_config(page_title="Bangalore House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction")
st.write("Predict house price using Machine Learning")

# UI Inputs
location = st.selectbox("📍 Location", locations)
sqft = st.number_input("📐 Total Square Feet", min_value=300)
bath = st.number_input("🛁 Number of Bathrooms", min_value=1, max_value=10)
bhk = st.number_input("🛏️ Number of BHK", min_value=1, max_value=10)

def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    return round(model.predict([x])[0], 2)

# Button
if st.button("🔮 Predict Price"):
    price = predict_price(location, sqft, bath, bhk)
    st.success(f"💰 Estimated Price: ₹ {price} Lakhs")
