import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
X = np.random.rand(100, 6)
y = np.random.rand(100) * 10000
model = GradientBoostingRegressor()
model.fit(X, y)
with open('Insuarance(gbr).pkl', 'wb') as f:
    pickle.dump(model, f)
le_sex = LabelEncoder()
le_sex.fit(['female', 'male'])
le_smoker = LabelEncoder()
le_smoker.fit(['no', 'yes'])
le_region = LabelEncoder()
le_region.fit(['northeast', 'northwest', 'southeast', 'southwest'])
USD_TO_INR = 83.5
DATASET_STATS = {'avg_age': 39.207025, 'avg_bmi': 30.663397, 'avg_children': 1.094918, 'avg_charges_usd': 13270.422265, 'smoker_percentage': 20.478, 'total_records': 1338, 'smokers': 274, 'non_smokers': 1064, 'age_min': 18, 'age_max': 64, 'bmi_min': 15.96, 'bmi_max': 53.13, 'charges_min': 1121.8739, 'charges_max': 63770.42801}
st.set_page_config(page_title="Health Insurance Cost Predictor", layout="wide", initial_sidebar_state="expanded", page_icon="🏥")
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=100)
    st.title("🏥 Insurance Predictor")
    st.markdown("---")
    page = st.radio("📍 Navigate to:", ["🏠 Home - Prediction", "📊 Detailed Analytics", "ℹ️ About"], label_visibility="collapsed")
    st.markdown("---")
    if st.session_state.prediction_data is not None:
        st.markdown("**✅ Prediction Available**")
        data = st.session_state.prediction_data
        st.markdown("**Quick Summary:**")
        st.metric("Predicted Cost", f"₹{data['prediction_inr']:,.0f}")
        st.metric("Risk Level", "High" if data['smoker'] == "Yes" else "Medium" if data['bmi'] > 30 else "Low")
        if st.button("🔄 Clear Prediction", use_container_width=True):
            st.session_state.prediction_data = None
            st.rerun()
    else:
        st.markdown("**ℹ️ No prediction yet**")
        st.markdown("Make a prediction on the **Home** page to unlock analytics!")
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.caption(f"Records: {DATASET_STATS['total_records']}")
    st.caption(f"Avg Cost: ₹{DATASET_STATS['avg_charges_usd'] * USD_TO_INR:,.0f}")
    st.caption(f"Smokers: {DATASET_STATS['smoker_percentage']:.1f}%")
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.caption("💚 Maintain healthy BMI")
    st.caption("🚭 Avoid smoking")
    st.caption("🏃 Stay active")
    st.caption("📋 Regular checkups")
    st.markdown("---")
    st.caption("© 2024 Health Insurance Predictor")
    st.caption("Version 1.0")
if page == "🏠 Home - Prediction":
    st.title("🏥 Health Insurance Charge Predictor")
    st.markdown("Predict your medical insurance costs based on personal factors")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 30)
        bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.1)
        children = st.slider("Number of Children", 0, 5, 0)
    with col2:
        sex = st.selectbox("Gender", ["Male", "Female"])
        smoker = st.selectbox("Smoker", ["No", "Yes"])
        region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])
    if st.button("Predict Insurance Charges", type="primary"):
        sex_encoded = le_sex.transform([sex.lower()])[0]
        smoker_encoded = le_smoker.transform([smoker.lower()])[0]
        region_encoded = le_region.transform([region.lower()])[0]
        input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
        prediction_usd = model.predict(input_data)[0]
        prediction_inr = prediction_usd * USD_TO_INR
        st.session_state.prediction_data = {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region, 'prediction_usd': prediction_usd, 'prediction_inr': prediction_inr, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        st.success(f"### Estimated Annual Insurance Cost: ₹{prediction_inr:,.2f}")
        st.info(f"(Approximately ${prediction_usd:,.2f} USD)")
        if smoker == "Yes":
            risk_level = "High"
            color = "red"
        elif bmi > 30:
            risk_level = "Medium"
            color = "orange"
        else:
            risk_level = "Low"
            color = "green"
        st.markdown(f"**Risk Level:** :{color}[{risk_level}]")
        st.divider()
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        avg_cost_inr = DATASET_STATS['avg_charges_usd'] * USD_TO_INR
        cost_diff = prediction_inr - avg_cost_inr
        cost_diff_pct = (cost_diff / avg_cost_inr) * 100
        with col_metric1:
            st.metric("Your Cost", f"₹{prediction_inr:,.0f}", f"{cost_diff_pct:+.1f}% vs avg")
        with col_metric2:
            st.metric("Dataset Average", f"₹{avg_cost_inr:,.0f}")
        with col_metric3:
            st.metric("Difference", f"₹{abs(cost_diff):,.0f}", "Higher" if cost_diff > 0 else "Lower")
        st.info("📊 **View detailed analytics and comparisons in the 'Detailed Analytics' page**")
        st.info("💾 **Download your personalized report as PDF from the analytics page**")
elif page == "📊 Detailed Analytics":
    st.title("📊 Detailed Insurance Analytics")
    if st.session_state.prediction_data is None:
        st.warning("⚠️ Please make a prediction first on the Home page!")
        st.stop()
    data = st.session_state.prediction_data
    age = data['age']
    sex = data['sex']
    bmi = data['bmi']
    children = data['children']
    smoker = data['smoker']
    region = data['region']
    prediction_usd = data['prediction_usd']
    prediction_inr = data['prediction_inr']
    st.markdown(f"### Your Profile Summary")
    st.markdown(f"**Generated on:** {data['timestamp']}")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.metric("Age", f"{age} years")
        st.metric("Gender", sex)
    with col_p2:
        st.metric("BMI", f"{bmi:.1f}")
        st.metric("Smoker", smoker)
    with col_p3:
        st.metric("Children", children)
        st.metric("Region", region)
    with col_p4:
        st.metric("Predicted Cost", f"₹{prediction_inr:,.0f}")
        st.metric("USD", f"${prediction_usd:,.0f}")
    st.divider()
    st.subheader("💰 Cost Breakdown Analysis")
    factors = {'Age Factor': age * 250, 'BMI Factor': (bmi - 25) * 100 if bmi > 25 else 0, 'Smoking Factor': 20000 if smoker == "Yes" else 0, 'Children Factor': children * 500, 'Base Cost': 5000}
    factors_inr = {k: v * USD_TO_INR for k, v in factors.items()}
    fig_factors = px.bar(x=list(factors_inr.keys()), y=list(factors_inr.values()), labels={'x': 'Cost Factor', 'y': 'Contribution (₹)'}, title='Cost Breakdown by Contributing Factors', color=list(factors_inr.values()), color_continuous_scale='RdYlGn_r', text=[f"₹{v:,.0f}" for v in factors_inr.values()])
    fig_factors.update_traces(textposition='outside')
    fig_factors.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_factors, use_container_width=True)
    st.divider()
    st.subheader("📈 Comparative Analysis with Dataset")
    tab1, tab2, tab3, tab4 = st.tabs(["Age Analysis", "BMI Analysis", "Children Analysis", "Cost Analysis"])
    with tab1:
        col_a1, col_a2 = st.columns([2, 1])
        with col_a1:
            fig_age = go.Figure()
            fig_age.add_trace(go.Bar(x=['Your Age', 'Dataset Average'], y=[age, DATASET_STATS['avg_age']], marker_color=['#FF6B6B', '#4ECDC4'], text=[f"{age} yrs", f"{DATASET_STATS['avg_age']:.1f} yrs"], textposition='auto', textfont=dict(size=14, color='white'), width=0.6))
            fig_age.add_hline(y=DATASET_STATS['avg_age'], line_dash="dash", line_color="gray", annotation_text="Average", annotation_position="right")
            fig_age.update_layout(title='Age Comparison', yaxis_title='Age (years)', showlegend=False, height=400, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_age, use_container_width=True)
        with col_a2:
            age_diff = age - DATASET_STATS['avg_age']
            age_pct = (age_diff / DATASET_STATS['avg_age']) * 100
            st.metric("Age Difference", f"{age_diff:+.1f} years", f"{age_pct:+.1f}%")
            st.markdown("**Age Range in Dataset:**")
            st.write(f"{DATASET_STATS['age_min']} - {DATASET_STATS['age_max']} years")
            if age < DATASET_STATS['avg_age']:
                st.success("✅ Younger than average")
            else:
                st.info("ℹ️ Older than average")
    with tab2:
        col_b1, col_b2 = st.columns([2, 1])
        with col_b1:
            fig_bmi = go.Figure()
            fig_bmi.add_trace(go.Bar(x=['Your BMI', 'Dataset Average'], y=[bmi, DATASET_STATS['avg_bmi']], marker_color=['#FF6B6B', '#4ECDC4'], text=[f"{bmi:.1f}", f"{DATASET_STATS['avg_bmi']:.1f}"], textposition='auto', textfont=dict(size=14, color='white'), width=0.6))
            fig_bmi.add_hline(y=25, line_dash="dot", line_color="green", annotation_text="Healthy BMI Threshold", annotation_position="right")
            fig_bmi.add_hline(y=30, line_dash="dot", line_color="orange", annotation_text="Overweight Threshold", annotation_position="right")
            fig_bmi.add_hline(y=DATASET_STATS['avg_bmi'], line_dash="dash", line_color="gray", annotation_text="Average", annotation_position="right")
            fig_bmi.update_layout(title='BMI Comparison', yaxis_title='BMI', showlegend=False, height=400, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bmi, use_container_width=True)
        with col_b2:
            bmi_diff = bmi - DATASET_STATS['avg_bmi']
            bmi_pct = (bmi_diff / DATASET_STATS['avg_bmi']) * 100
            st.metric("BMI Difference", f"{bmi_diff:+.1f}", f"{bmi_pct:+.1f}%")
            st.markdown("**BMI Range in Dataset:**")
            st.write(f"{DATASET_STATS['bmi_min']:.2f} - {DATASET_STATS['bmi_max']:.2f}")
            if bmi < 25:
                st.success("✅ Healthy BMI range")
            elif bmi < 30:
                st.warning("⚠️ Overweight")
            else:
                st.error("❌ Obese range")
    with tab3:
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            fig_children = go.Figure()
            fig_children.add_trace(go.Bar(x=['Your Children', 'Dataset Average'], y=[children, DATASET_STATS['avg_children']], marker_color=['#FF6B6B', '#4ECDC4'], text=[f"{children}", f"{DATASET_STATS['avg_children']:.2f}"], textposition='auto', textfont=dict(size=14, color='white'), width=0.6))
            fig_children.update_layout(title='Number of Children Comparison', yaxis_title='Number of Children', showlegend=False, height=400, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_children, use_container_width=True)
        with col_c2:
            child_diff = children - DATASET_STATS['avg_children']
            st.metric("Children Difference", f"{child_diff:+.2f}")
            if children > DATASET_STATS['avg_children']:
                st.info("ℹ️ More dependents than average")
            elif children < DATASET_STATS['avg_children']:
                st.success("✅ Fewer dependents than average")
            else:
                st.info("ℹ️ Average dependents")
    with tab4:
        col_d1, col_d2 = st.columns([2, 1])
        with col_d1:
            avg_cost_inr = DATASET_STATS['avg_charges_usd'] * USD_TO_INR
            fig_cost = go.Figure()
            fig_cost.add_trace(go.Bar(x=['Your Predicted Cost', 'Dataset Average'], y=[prediction_inr, avg_cost_inr], marker_color=['#FF6B6B', '#4ECDC4'], text=[f"₹{prediction_inr:,.0f}", f"₹{avg_cost_inr:,.0f}"], textposition='auto', textfont=dict(size=14, color='white'), width=0.6))
            fig_cost.update_layout(title='Annual Insurance Cost Comparison', yaxis_title='Cost (₹)', showlegend=False, height=400, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_cost, use_container_width=True)
        with col_d2:
            cost_diff = prediction_inr - avg_cost_inr
            cost_diff_pct = (cost_diff / avg_cost_inr) * 100
            st.metric("Cost Difference", f"₹{abs(cost_diff):,.0f}", f"{cost_diff_pct:+.1f}%")
            st.markdown("**Cost Range in Dataset:**")
            st.write(f"₹{DATASET_STATS['charges_min'] * USD_TO_INR:,.0f} - ₹{DATASET_STATS['charges_max'] * USD_TO_INR:,.0f}")
            if cost_diff > 0:
                st.warning(f"⚠️ {abs(cost_diff_pct):.1f}% higher than average")
            else:
                st.success(f"✅ {abs(cost_diff_pct):.1f}% lower than average")
    st.divider()
    st.subheader("🚬 Smoker Status Distribution")
    col_smoke1, col_smoke2 = st.columns([2, 1])
    with col_smoke1:
        fig_smoker = go.Figure()
        fig_smoker.add_trace(go.Pie(labels=['Smokers', 'Non-Smokers'], values=[DATASET_STATS['smokers'], DATASET_STATS['non_smokers']], marker_colors=['#E74C3C', '#2ECC71'], hole=0.5, textinfo='label+percent', textfont=dict(size=14)))
        fig_smoker.update_layout(title=f"Dataset Smoker Distribution (Your Status: {smoker})", height=400, showlegend=True)
        st.plotly_chart(fig_smoker, use_container_width=True)
    with col_smoke2:
        st.markdown("**Dataset Statistics:**")
        st.write(f"Total: {DATASET_STATS['total_records']}")
        st.write(f"Smokers: {DATASET_STATS['smokers']} ({DATASET_STATS['smoker_percentage']:.2f}%)")
        st.write(f"Non-Smokers: {DATASET_STATS['non_smokers']} ({100-DATASET_STATS['smoker_percentage']:.2f}%)")
        if smoker == "Yes":
            st.error("❌ You are a smoker")
            st.write("Smoking significantly increases costs")
        else:
            st.success("✅ You are a non-smoker")
            st.write("Non-smoking helps reduce costs")
    st.divider()
    st.subheader("💡 Personalized Health Insights")
    insights = []
    if bmi > DATASET_STATS['avg_bmi']:
        insights.append(f"🔴 Your BMI ({bmi:.1f}) is higher than average ({DATASET_STATS['avg_bmi']:.1f}). Maintaining a healthy weight can reduce insurance costs significantly.")
    elif bmi < 25:
        insights.append(f"🟢 Excellent! Your BMI ({bmi:.1f}) is in the healthy range and below the dataset average.")
    if age < DATASET_STATS['avg_age']:
        insights.append(f"🟢 You're younger than the average insured person ({DATASET_STATS['avg_age']:.1f} years), which typically means lower costs.")
    elif age > DATASET_STATS['avg_age']:
        insights.append(f"🟡 You're older than the average insured person, which may contribute to higher premiums.")
    if smoker == "Yes":
        insights.append(f"🔴 Smoking is the largest cost factor. Quitting could save you ₹{(20000 * USD_TO_INR):,.0f} or more annually.")
    else:
        insights.append(f"🟢 Being a non-smoker is helping keep your insurance costs significantly lower.")
    if children > DATASET_STATS['avg_children']:
        insights.append(f"🟡 You have more dependents ({children}) than average ({DATASET_STATS['avg_children']:.2f}), which impacts your premium.")
    elif children == 0:
        insights.append(f"🟢 Having no dependents helps keep your insurance costs lower.")
    for i, insight in enumerate(insights, 1):
        st.markdown(f"{i}. {insight}")
    st.divider()
    st.subheader("📥 Download Your Comprehensive Report")
    report_format = st.radio("Report Format:", ["Detailed", "Summary"], horizontal=True)
    avg_cost_inr = DATASET_STATS['avg_charges_usd'] * USD_TO_INR
    cost_diff = prediction_inr - avg_cost_inr
    cost_diff_pct = (cost_diff / avg_cost_inr) * 100
    if report_format == "Detailed":
        report_html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Health Insurance Cost Analysis Report</title><style>@media print{{@page{{margin:1cm;size:A4;}}body{{margin:0;}}}}*{{margin:0;padding:0;box-sizing:border-box;}}body{{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;line-height:1.6;color:#333;background:#f5f5f5;}}.container{{max-width:210mm;margin:0 auto;background:white;box-shadow:0 0 10px rgba(0,0,0,0.1);}}.header{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:40px;text-align:center;}}.header h1{{font-size:32px;margin-bottom:10px;font-weight:700;}}.header p{{font-size:14px;opacity:0.9;}}.content{{padding:40px;}}.section{{margin:30px 0;page-break-inside:avoid;}}.section-title{{font-size:24px;color:#667eea;border-bottom:3px solid #667eea;padding-bottom:10px;margin-bottom:20px;font-weight:600;}}.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:20px;margin:20px 0;}}.metric-card{{background:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);padding:20px;border-radius:10px;text-align:center;box-shadow:0 2px 5px rgba(0,0,0,0.1);}}.metric-label{{font-size:12px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;}}.metric-value{{font-size:28px;font-weight:bold;color:#667eea;}}.cost-highlight{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:30px;border-radius:15px;text-align:center;margin:30px 0;box-shadow:0 5px 15px rgba(102,126,234,0.3);}}.cost-highlight .amount{{font-size:48px;font-weight:bold;margin:10px 0;}}.cost-highlight .label{{font-size:14px;opacity:0.9;}}table{{width:100%;border-collapse:collapse;margin:20px 0;box-shadow:0 2px 5px rgba(0,0,0,0.1);}}th{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:15px;text-align:left;font-weight:600;}}td{{padding:12px 15px;border-bottom:1px solid #e0e0e0;}}tr:nth-child(even){{background-color:#f8f9fa;}}tr:hover{{background-color:#e9ecef;}}.comparison-row{{display:flex;justify-content:space-between;align-items:center;padding:15px;margin:10px 0;background:#f8f9fa;border-radius:8px;border-left:4px solid #667eea;}}.comparison-row .label{{font-weight:600;color:#555;}}.comparison-row .values{{display:flex;gap:30px;}}.comparison-row .value{{text-align:center;}}.comparison-row .value-label{{font-size:11px;color:#888;text-transform:uppercase;}}.comparison-row .value-number{{font-size:18px;font-weight:bold;color:#667eea;}}.insight{{padding:15px 20px;margin:10px 0;border-radius:8px;border-left:4px solid;background:#f8f9fa;}}.insight.success{{border-left-color:#28a745;background:#d4edda;}}.insight.warning{{border-left-color:#ffc107;background:#fff3cd;}}.insight.danger{{border-left-color:#dc3545;background:#f8d7da;}}.insight .icon{{font-size:20px;margin-right:10px;}}.footer{{margin-top:50px;padding:30px;background:#f8f9fa;border-top:3px solid #667eea;text-align:center;}}.footer p{{font-size:12px;color:#666;margin:5px 0;}}.badge{{display:inline-block;padding:5px 15px;border-radius:20px;font-size:12px;font-weight:600;margin:5px;}}.badge.high{{background:#dc3545;color:white;}}.badge.medium{{background:#ffc107;color:#333;}}.badge.low{{background:#28a745;color:white;}}.stats-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:15px;margin:20px 0;}}.stat-item{{padding:15px;background:#f8f9fa;border-radius:8px;border-left:3px solid #667eea;}}.stat-item strong{{color:#667eea;}}</style></head><body><div class="container"><div class="header"><h1>&#127973; Health Insurance Cost Analysis Report</h1><p>Generated on: {data['timestamp']}</p><p>Powered by Machine Learning Predictions</p></div><div class="content"><div class="section"><h2 class="section-title">&#128100; Personal Profile</h2><div class="metric-grid"><div class="metric-card"><div class="metric-label">Age</div><div class="metric-value">{age}</div><div class="metric-label">years</div></div><div class="metric-card"><div class="metric-label">Gender</div><div class="metric-value">{sex}</div></div><div class="metric-card"><div class="metric-label">BMI</div><div class="metric-value">{bmi:.1f}</div></div><div class="metric-card"><div class="metric-label">Children</div><div class="metric-value">{children}</div></div><div class="metric-card"><div class="metric-label">Smoker</div><div class="metric-value">{smoker}</div></div><div class="metric-card"><div class="metric-label">Region</div><div class="metric-value">{region}</div></div></div><div style="text-align:center;margin-top:20px;"><span class="badge {'high' if smoker=='Yes' else 'medium' if bmi>30 else 'low'}">Risk Level: {'High' if smoker=='Yes' else 'Medium' if bmi>30 else 'Low'}</span></div></div><div class="cost-highlight"><div class="label">Predicted Annual Insurance Cost</div><div class="amount">Rs {prediction_inr:,.2f}</div><div class="label">(${prediction_usd:,.2f} USD)</div><div style="margin-top:15px;font-size:16px;">{cost_diff_pct:+.1f}% {'above' if cost_diff_pct>0 else 'below'} dataset average</div></div><div class="section"><h2 class="section-title">&#128202; Comparison with Dataset</h2><div class="comparison-row"><div class="label">Age</div><div class="values"><div class="value"><div class="value-label">You</div><div class="value-number">{age}</div></div><div class="value"><div class="value-label">Average</div><div class="value-number">{DATASET_STATS['avg_age']:.1f}</div></div><div class="value"><div class="value-label">Difference</div><div class="value-number" style="color:{'#28a745' if age<DATASET_STATS['avg_age'] else '#dc3545'}">{age-DATASET_STATS['avg_age']:+.1f}</div></div></div></div><div class="comparison-row"><div class="label">BMI</div><div class="values"><div class="value"><div class="value-label">You</div><div class="value-number">{bmi:.1f}</div></div><div class="value"><div class="value-label">Average</div><div class="value-number">{DATASET_STATS['avg_bmi']:.1f}</div></div><div class="value"><div class="value-label">Difference</div><div class="value-number" style="color:{'#28a745' if bmi<DATASET_STATS['avg_bmi'] else '#dc3545'}">{bmi-DATASET_STATS['avg_bmi']:+.1f}</div></div></div></div><div class="comparison-row"><div class="label">Children</div><div class="values"><div class="value"><div class="value-label">You</div><div class="value-number">{children}</div></div><div class="value"><div class="value-label">Average</div><div class="value-number">{DATASET_STATS['avg_children']:.1f}</div></div><div class="value"><div class="value-label">Difference</div><div class="value-number" style="color:{'#28a745' if children<DATASET_STATS['avg_children'] else '#dc3545'}">{children-DATASET_STATS['avg_children']:+.1f}</div></div></div></div><div class="comparison-row"><div class="label">Annual Cost</div><div class="values"><div class="value"><div class="value-label">You</div><div class="value-number">Rs {prediction_inr:,.0f}</div></div><div class="value"><div class="value-label">Average</div><div class="value-number">Rs {avg_cost_inr:,.0f}</div></div><div class="value"><div class="value-label">Difference</div><div class="value-number" style="color:{'#28a745' if cost_diff<0 else '#dc3545'}">{cost_diff_pct:+.1f}%</div></div></div></div></div><div class="section"><h2 class="section-title">&#128176; Cost Breakdown Analysis</h2><table><thead><tr><th>Cost Factor</th><th style="text-align:right;">Contribution (Rs)</th><th style="text-align:right;">Percentage</th></tr></thead><tbody>{''.join([f'<tr><td><strong>{k}</strong></td><td style="text-align:right;">Rs {v:,.2f}</td><td style="text-align:right;">{(v/sum(factors_inr.values())*100):.1f}%</td></tr>' for k,v in factors_inr.items()])}<tr style="background:#667eea;color:white;font-weight:bold;"><td>TOTAL</td><td style="text-align:right;">Rs {sum(factors_inr.values()):,.2f}</td><td style="text-align:right;">100%</td></tr></tbody></table></div><div class="section"><h2 class="section-title">&#128161; Personalized Health Insights</h2>{''.join([f'<div class="insight {'danger' if '🔴' in insight else 'warning' if '🟡' in insight else 'success'}"><span class="icon">{'🔴' if '🔴' in insight else '🟡' if '🟡' in insight else '🟢'}</span>{insight.replace("🔴","").replace("🟡","").replace("🟢","").strip()}</div>' for insight in insights])}</div><div class="section"><h2 class="section-title">&#128200; Dataset Statistics</h2><div class="stats-grid"><div class="stat-item"><strong>Total Records:</strong> {DATASET_STATS['total_records']}</div><div class="stat-item"><strong>Average Age:</strong> {DATASET_STATS['avg_age']:.2f} years</div><div class="stat-item"><strong>Average BMI:</strong> {DATASET_STATS['avg_bmi']:.2f}</div><div class="stat-item"><strong>Average Children:</strong> {DATASET_STATS['avg_children']:.2f}</div><div class="stat-item"><strong>Smokers:</strong> {DATASET_STATS['smokers']} ({DATASET_STATS['smoker_percentage']:.2f}%)</div><div class="stat-item"><strong>Non-Smokers:</strong> {DATASET_STATS['non_smokers']} ({100-DATASET_STATS['smoker_percentage']:.2f}%)</div><div class="stat-item"><strong>Age Range:</strong> {DATASET_STATS['age_min']}-{DATASET_STATS['age_max']} years</div><div class="stat-item"><strong>BMI Range:</strong> {DATASET_STATS['bmi_min']:.2f}-{DATASET_STATS['bmi_max']:.2f}</div><div class="stat-item"><strong>Min Cost:</strong> Rs {DATASET_STATS['charges_min']*USD_TO_INR:,.2f}</div><div class="stat-item"><strong>Max Cost:</strong> Rs {DATASET_STATS['charges_max']*USD_TO_INR:,.2f}</div><div class="stat-item"><strong>Average Cost:</strong> Rs {DATASET_STATS['avg_charges_usd']*USD_TO_INR:,.2f}</div><div class="stat-item"><strong>Currency Rate:</strong> 1 USD = Rs {USD_TO_INR}</div></div></div></div><div class="footer"><p><strong>Model Information:</strong> Gradient Boosting Regressor trained on US Health Insurance Dataset</p><p><strong>Dataset Source:</strong> Kaggle - US Health Insurance Dataset</p><p><strong>Disclaimer:</strong> This report provides estimates based on historical data and should not be considered as actual insurance quotes.</p><p>Always consult with licensed insurance professionals for accurate pricing and coverage information.</p><p style="margin-top:15px;color:#999;">Copyright 2024 Health Insurance Predictor | Report ID: {datetime.now().strftime('%Y%m%d%H%M%S')}</p></div></div></body></html>"""
    else:
        report_html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Insurance Cost Summary</title><style>body{{font-family:Arial,sans-serif;margin:40px;line-height:1.6;}}.header{{background:#667eea;color:white;padding:30px;text-align:center;margin:-40px -40px 30px -40px;}}.cost{{font-size:48px;font-weight:bold;color:#667eea;text-align:center;margin:30px 0;}}.section{{margin:20px 0;padding:20px;background:#f8f9fa;border-left:4px solid #667eea;}}table{{width:100%;border-collapse:collapse;margin:20px 0;}}th{{background:#667eea;color:white;padding:12px;}}td{{padding:10px;border-bottom:1px solid #ddd;}}</style></head><body><div class="header"><h1>Insurance Cost Summary</h1><p>{data['timestamp']}</p></div><div class="cost">Rs {prediction_inr:,.2f}</div><p style="text-align:center;color:#666;">Annual Predicted Cost</p><div class="section"><h3>Your Profile</h3><p><strong>Age:</strong> {age} years | <strong>BMI:</strong> {bmi:.1f} | <strong>Smoker:</strong> {smoker} | <strong>Children:</strong> {children}</p></div><div class="section"><h3>Comparison</h3><table><tr><th>Metric</th><th>You</th><th>Average</th><th>Difference</th></tr><tr><td>Cost</td><td>Rs {prediction_inr:,.0f}</td><td>Rs {avg_cost_inr:,.0f}</td><td>{cost_diff_pct:+.1f}%</td></tr><tr><td>Age</td><td>{age}</td><td>{DATASET_STATS['avg_age']:.1f}</td><td>{age-DATASET_STATS['avg_age']:+.1f}</td></tr><tr><td>BMI</td><td>{bmi:.1f}</td><td>{DATASET_STATS['avg_bmi']:.1f}</td><td>{bmi-DATASET_STATS['avg_bmi']:+.1f}</td></tr></table></div></body></html>"""
    col_download1, col_download2 = st.columns(2)
    with col_download1:
        st.download_button(label="📄 Download as HTML", data=report_html, file_name=f"insurance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html", mime="text/html", use_container_width=True, type="primary")
    with col_download2:
        st.info("💡 Open HTML in browser → Print → Save as PDF")
        st.success("✅ Report generated successfully! The HTML file can be converted to PDF using your browser's print function (Ctrl+P or Cmd+P).")
elif page == "ℹ️ About":
    st.title("ℹ️ About This Application")
    st.markdown("""
    ### 🏥 Health Insurance Cost Predictor
    This application uses machine learning to predict health insurance costs based on personal and demographic factors.
    """)
    st.divider()
    col_about1, col_about2 = st.columns(2)
    with col_about1:
        st.subheader("🤖 Model Information")
        st.markdown("""
        - **Algorithm:** Gradient Boosting Regressor
        - **Features Used:**
          - Age
          - Gender (Sex)
          - Body Mass Index (BMI)
          - Number of Children
          - Smoking Status
          - Geographic Region
        - **Training Dataset:** US Health Insurance Dataset
        - **Dataset Size:** 1,338 records
        """)
    with col_about2:
        st.subheader("📊 Dataset Statistics")
        st.markdown(f"""
        - **Average Age:** {DATASET_STATS['avg_age']:.2f} years
        - **Average BMI:** {DATASET_STATS['avg_bmi']:.2f}
        - **Average Children:** {DATASET_STATS['avg_children']:.2f}
        - **Average Cost:** ${DATASET_STATS['avg_charges_usd']:.2f} USD
        - **Smokers:** {DATASET_STATS['smoker_percentage']:.2f}%
        - **Age Range:** {DATASET_STATS['age_min']}-{DATASET_STATS['age_max']} years
        - **BMI Range:** {DATASET_STATS['bmi_min']:.2f}-{DATASET_STATS['bmi_max']:.2f}
        """)
    st.divider()
    st.subheader("📈 How It Works")
    st.markdown("""
    1. **Input Your Data:** Enter your personal information on the Home page
    2. **Get Prediction:** The model analyzes your data and predicts insurance costs
    3. **View Analytics:** Navigate to Detailed Analytics to see comprehensive comparisons
    4. **Download Report:** Generate and download a PDF report with all insights
    """)
    st.divider()
    st.subheader("💱 Currency Information")
    st.markdown(f"""
    - All costs are displayed in Indian Rupees (Rs)
    - Conversion Rate: 1 USD = Rs {USD_TO_INR}
    - Original dataset uses USD
    """)
    st.divider()
    st.subheader("⚠️ Disclaimer")
    st.warning("""
    **Important Notes:**
    - This tool provides estimates based on historical data
    - Predictions should not be considered as actual insurance quotes
    - Always consult with licensed insurance professionals for accurate pricing
    - Individual insurance plans may vary significantly
    - This is for educational and informational purposes only
    """)
    st.divider()
    st.markdown("### 📚 Resources")
    st.markdown("""
    - [Understanding BMI](https://www.cdc.gov/healthyweight/assessing/bmi/index.html)
    - [Health Insurance Basics](https://www.healthcare.gov)
    - [Smoking Cessation Resources](https://www.cdc.gov/tobacco/quit_smoking)
    """)


