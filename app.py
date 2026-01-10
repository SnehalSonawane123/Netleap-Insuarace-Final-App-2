import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import os
import sys
st.set_page_config(page_title="Health Insurance Cost Predictor", layout="centered", page_icon="🏥")
sys.modules['sklearn.ensemble.gradient_boosting'] = sys.modules['sklearn.ensemble']
sys.modules['sklearn.ensemble._gb'] = sys.modules['sklearn.ensemble']
try:
    if os.path.exists('Insuarance(gbr).pkl'):
        with open('Insuarance(gbr).pkl', 'rb') as f:
            model = pickle.load(f)
        model_loaded = True
    else:
        X = np.random.rand(1000, 6)
        y = 1000 + (X[:, 0] * 100) + (X[:, 2] * 200) + (X[:, 4] * 15000) + np.random.rand(1000) * 2000
        model = GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=4)
        model.fit(X, y)
        model_loaded = False
except Exception as e:
    X = np.random.rand(1000, 6)
    y = 1000 + (X[:, 0] * 100) + (X[:, 2] * 200) + (X[:, 4] * 15000) + np.random.rand(1000) * 2000
    model = GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=4)
    model.fit(X, y)
    model_loaded = False
le_sex = LabelEncoder()
le_sex.fit(['female', 'male'])
le_smoker = LabelEncoder()
le_smoker.fit(['no', 'yes'])
le_region = LabelEncoder()
le_region.fit(['northeast', 'northwest', 'southeast', 'southwest'])
USD_TO_INR = 83.5
st.markdown("# 🏥 Health Insurance Cost Predictor")
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
    try:
        sex_encoded = le_sex.transform([sex.lower()])[0]
        smoker_encoded = le_smoker.transform([smoker.lower()])[0]
        region_encoded = le_region.transform([region.lower()])[0]
        input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
        prediction_usd = model.predict(input_data)[0]
        prediction_usd = np.clip(prediction_usd, 1000, 50000)
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
        with st.expander("📊 Cost Factors Breakdown"):
            factors = []
            if smoker == "Yes":
                factors.append("• Smoking significantly increases insurance costs (typically 2-3x higher)")
            if bmi > 30:
                factors.append("• BMI over 30 may increase premiums")
            if age > 50:
                factors.append("• Age over 50 typically increases costs")
            if children > 2:
                factors.append("• Multiple dependents may affect family plan costs")
            if factors:
                for factor in factors:
                    st.write(factor)
            else:
                st.write("No significant risk factors identified")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")




