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
# income is measured in thousands of dollars; default value should respect the min_value
# previous min_value of 20000 caused a StreamlitValueBelowMinError when value=50
income = st.number_input("Annual Income (k$)", min_value=0, value=50)
Total_spend = st.number_input("Total Spending (sum of purchases)", min_value=0, max_value=1000, value=50)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, value=10)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=5)
num_web_visits = st.number_input("Number of Web Visits", min_value=0, max_value=50, value=20)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)


# construct DataFrame using the exact feature names that were used when
# fitting the scaler and training kmeans (see Analysis_Model.ipynb):
# ['Age','Income','Total_spend','NumWebPurchases','NumStorePurchases',
#  'NumWebVisitsMonth','Recency']
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_spend": [Total_spend],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]

    # show the result prominently
    st.success(f"🎯 Predicted Customer Segment: **Cluster {cluster}**")

    # detailed description of all clusters using emojis and markdown list
    st.markdown("""
**Cluster breakdown:**

- 🥇 **Cluster 0**: High income, frequent web visitors, high spending — *Premium Customers*
- 😊 **Cluster 1**: Moderate income & spending — *Average Customers*
- 🛒 **Cluster 2**: High web & store purchases — *Digital Buyers*
- 💸 **Cluster 3**: Low income, low spending — *Budget Customers*
- 🤝 **Cluster 4**: Moderate income, high spending — *Loyal Customers*
- 💤 **Cluster 5**: Low recency (inactive) — *Inactive Customers*
""")
             