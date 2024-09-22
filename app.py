import streamlit as st
import numpy as np
import pickle

st.title('House Price Predictor')

# Load the model
with open('house_model.pkl', 'rb') as file:
    model = pickle.load(file)

# User inputs
Area = float(st.number_input('Enter Area'))  # Convert to float
Room = int(st.number_input('Enter Room'))     # Convert to int
Lon = float(st.number_input('Enter Lon'))      # Convert to float
Lat = float(st.number_input('Enter Lat'))      # Convert to float

# Prepare the data for prediction
data = np.array([[Area, Room, Lon, Lat]])  # Create a 2D array for a single sample

# Make prediction
pred = model.predict(data)

# Show prediction
st.write(f'Predicted House Price: {pred[0]:,.2f}')  # Format the output for better readability
