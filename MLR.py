import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import streamlit as st

# Ensure the directory for saving models exists
MODEL_DIR = 'models'

# Define feature and target columns
feature_columns = ['A(410)', 'B(435)', 'C(460)', 'D(485)', 'E(510)', 'F(535)', 'G(560)', 'H(585)', 'R(610)',
                   'I(645)', 'S(680)', 'J(705)', 'T(730)', 'U(760)', 'V(810)', 'W(860)', 'K(900)', 'L(940)']
target_columns = ['pH', 'EC (dS/m)', 'OC (%)', 'P (kg/ha)', 'K (kg/ha)', 'Ca (meq/100g)', 'Mg (meq/100g)', 
                  'S (ppm)', 'Fe (ppm)', 'Mn (ppm)', 'Cu (ppm)', 'Zn (ppm)', 'B (ppm)']

# Function to predict based on user input wavebands
def predict_property(user_input_wavebands, best_wavebands, best_model):
    # Ensure the user input matches the number of best wavebands
    if len(user_input_wavebands) != len(best_wavebands):
        raise ValueError(f"Expected {len(best_wavebands)} wavebands, got {len(user_input_wavebands)}")
    
    # Create DataFrame for user input
    user_input_df = pd.DataFrame([user_input_wavebands], columns=best_wavebands)
    prediction = best_model.predict(user_input_df)
    return prediction

# Streamlit App
st.title('Soil Property Prediction')

# Select a property to predict
property_option = st.selectbox('Choose a property to predict:', target_columns)

if property_option:
    target_column = property_option

    # Load the model and wavebands for the selected property
    try:
        best_model = joblib.load(os.path.join(MODEL_DIR, f'{target_column.replace("/", "_")}_model.pkl'))
        best_wavebands = joblib.load(os.path.join(MODEL_DIR, f'{target_column.replace("/", "_")}_wavebands.pkl'))
    except FileNotFoundError:
        st.error(f"No saved model found for {target_column}.")
    else:
        st.write(f"Loaded model for predicting {target_column}.")

        # Get user input for waveband values
        user_input_wavebands = []
        for waveband in best_wavebands:
            value = st.number_input(f"Enter value for waveband {waveband}:", value=0.0)
            user_input_wavebands.append(value)

        # Predict and display the result
        if st.button('Predict'):
            prediction = predict_property(user_input_wavebands, best_wavebands, best_model)
            st.success(f"Predicted {target_column}: {prediction[0]}")

