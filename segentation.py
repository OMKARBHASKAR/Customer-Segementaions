import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Segmentation Analysis")
st.write("Enter customer data to predict their segment.")


age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income (k$)", min_value=20000, value=50)
Total_spend = st.number_input("Total Spending (sum of purchases)", min_value=0,max_value=1000, value=50)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, value=10)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0  , max_value =100, value=5)
num_web_visits = st.number_input("Number of Web Visits", min_value=0, max_value=50, value=20)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)


input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_spend": [Total_spend],
    "num_web_purchases": [num_web_purchases],
    "num_store_purchases": [num_store_purchases],
    "NumVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    
    st.write(f"Predicted Customer Segment: Cluster {cluster}")
    
    st.write("""Cluster 0: High Income, web visiters ,High Spending - Premium Customers"""
                """Cluster 1: Moderate Income, Moderate Spending - Average Customers"""
                """Cluster 2: High Web Purchases, High Store Purchases - Digital Buyers"""
                """Cluster 3: Low Income, Low Spending - Budget Customers"""
                """Cluster 4: Moderate Income, High Spending - Loyal Customers"""
                """Cluster 5: Low Recency, Inactive Customers""")
             