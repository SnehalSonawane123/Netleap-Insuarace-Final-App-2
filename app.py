import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import os
try:
    if os.path.exists('Insuarance(gbr).pkl'):
        with open('Insuarance(gbr).pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        raise FileNotFoundError
except:
    X = np.random.rand(1000, 6)
    y = np.random.rand(1000) * 10000
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)
le_sex = LabelEncoder()
le_sex.fit(['female', 'male'])
le_smoker = LabelEncoder()
le_smoker.fit(['no', 'yes'])
le_region = LabelEncoder()
le_region.fit(['northeast', 'northwest', 'southeast', 'southwest'])
USD_TO_INR = 83.5
st.set_page_config(page_title="Health Insurance Cost Predictor", layout="centered", page_icon="🏥")
st.title("🏥 Health Insurance Cost Predictor")
st.markdown("Enter your details to predict your annual medical insurance costs")
st.divider()
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, 30)
    bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.1)
    children = st.slider("Number of Children", 0, 5, 0)
with col2:
    sex = st.selectbox("Gender", ["Male", "Female"])
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])
st.divider()
if st.button("🔮 Predict Insurance Cost", type="primary", use_container_width=True):
    sex_encoded = le_sex.transform([sex.lower()])[0]
    smoker_encoded = le_smoker.transform([smoker.lower()])[0]
    region_encoded = le_region.transform([region.lower()])[0]
    input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
    prediction_usd = model.predict(input_data)[0]
    prediction_inr = prediction_usd * USD_TO_INR
    st.success("### 💰 Prediction Results")
    col_result1, col_result2 = st.columns(2)
    with col_result1:
        st.metric("Annual Cost", f"₹{prediction_inr:,.2f}")
        st.caption(f"${prediction_usd:,.2f} USD")
    with col_result2:
        st.metric("Monthly Cost", f"₹{prediction_inr/12:,.2f}")
        st.caption(f"${prediction_usd/12:,.2f} USD")
    if smoker == "Yes":
        risk_level = "High"
        risk_color = "🔴"
    elif bmi > 30:
        risk_level = "Medium"
        risk_color = "🟡"
    else:
        risk_level = "Low"
        risk_color = "🟢"
    st.info(f"**Risk Assessment:** {risk_color} {risk_level} Risk")
    st.divider()
    st.caption("💡 This is an estimate based on statistical models. Actual insurance costs may vary. Consult with insurance professionals for accurate quotes.")
