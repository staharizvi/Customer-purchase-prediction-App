import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the model and scaler
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to make predictions
def predict_purchases(features):
    # Prepare the input features
    features_df = pd.DataFrame([features], columns=[
        'Age', 'Gender', 'AnnualIncome', 'NumberOfPurchases',
        'ProductCategory', 'TimeSpentOnWebsite', 'LoyaltyProgram', 'DiscountsAvailed'
    ])
    
    # Scale the features
    features_scaled = scaler.transform(features_df)
    
    # Predict
    prediction = rf_model.predict(features_scaled)
    return prediction[0]

# Streamlit app
def main():
    st.title('Customer Purchase Prediction')

    st.write("""
    This app predicts whether a customer will make a purchase based on the provided features.
    Fill in the inputs below and click "Predict" to see the result.
    """)

    # Create two columns
    col1, col2 = st.columns(2)

    # Input fields in the first column
    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=30)
        annual_income = st.number_input('Annual Income', min_value=0, value=50000)
        number_of_purchases = st.number_input('Number of Purchases', min_value=0, value=1)
        time_spent_on_website = st.number_input('Time Spent on Website (minutes)', min_value=0, value=30)
    
    # Input fields in the second column
    with col2:
        gender = st.selectbox('Gender', options=['Male', 'Female'])
        product_category = st.selectbox('Product Category', options=['Electronics', 'Clothing', 'Groceries', 'Home'])
        loyalty_program = st.selectbox('Loyalty Program', options=['Yes', 'No'])
        discounts_availed = st.number_input('Discounts Availed', min_value=0, value=0)

    # Convert categorical data to numerical
    gender = 1 if gender == 'Male' else 0
    loyalty_program = 1 if loyalty_program == 'Yes' else 0
    product_category = {
        'Electronics': 0,
        'Clothing': 1,
        'Groceries': 2,
        'Home': 3
    }[product_category]

    features = [
        age, gender, annual_income, number_of_purchases,
        product_category, time_spent_on_website, loyalty_program, discounts_availed
    ]

    # Make prediction
    if st.button('Predict'):
        prediction = predict_purchases(features)
        st.write(f'Prediction: {"Purchased" if prediction == 1 else "Not Purchased"}')

if __name__ == '__main__':
    main()
