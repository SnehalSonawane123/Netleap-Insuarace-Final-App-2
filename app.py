import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
st.set_page_config(page_title="Health Insurance Cost Predictor", layout="wide", page_icon="🏥")
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
np.random.seed(42)
db_size = 5000
database = pd.DataFrame({
    'age': np.random.randint(18, 65, db_size),
    'sex': np.random.choice(['male', 'female'], db_size),
    'bmi': np.random.normal(28, 6, db_size).clip(15, 50),
    'children': np.random.choice([0, 1, 2, 3, 4, 5], db_size, p=[0.3, 0.25, 0.25, 0.1, 0.07, 0.03]),
    'smoker': np.random.choice(['no', 'yes'], db_size, p=[0.8, 0.2]),
    'region': np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], db_size)
})
le_sex = LabelEncoder()
le_sex.fit(['female', 'male'])
le_smoker = LabelEncoder()
le_smoker.fit(['no', 'yes'])
le_region = LabelEncoder()
le_region.fit(['northeast', 'northwest', 'southeast', 'southwest'])
database['sex_encoded'] = le_sex.transform(database['sex'])
database['smoker_encoded'] = le_smoker.transform(database['smoker'])
database['region_encoded'] = le_region.transform(database['region'])
X_db = database[['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 'region_encoded']].values
database['predicted_cost'] = model.predict(X_db)
USD_TO_INR = 83.5
def create_comparison_charts(user_data, prediction_usd, database):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('white')
    avg_cost_overall = database['predicted_cost'].mean()
    axes[0, 0].bar(['Your Cost', 'Average Cost'], [prediction_usd, avg_cost_overall], color=['#FF6B6B', '#4ECDC4'])
    axes[0, 0].set_ylabel('Cost (USD)', fontsize=10)
    axes[0, 0].set_title('Your Cost vs Database Average', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate([prediction_usd, avg_cost_overall]):
        axes[0, 0].text(i, v + 500, f'${v:,.0f}', ha='center', va='bottom', fontweight='bold')
    age_groups = pd.cut(database['age'], bins=[0, 30, 40, 50, 100], labels=['18-30', '31-40', '41-50', '51+'])
    age_avg = database.groupby(age_groups)['predicted_cost'].mean()
    user_age_group = pd.cut([user_data['age']], bins=[0, 30, 40, 50, 100], labels=['18-30', '31-40', '41-50', '51+'])[0]
    colors_age = ['#FF6B6B' if group == user_age_group else '#95E1D3' for group in age_avg.index]
    axes[0, 1].bar(age_avg.index, age_avg.values, color=colors_age)
    axes[0, 1].axhline(y=prediction_usd, color='red', linestyle='--', linewidth=2, label='Your Cost')
    axes[0, 1].set_ylabel('Average Cost (USD)', fontsize=10)
    axes[0, 1].set_title('Cost by Age Group', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    bmi_groups = pd.cut(database['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    bmi_avg = database.groupby(bmi_groups)['predicted_cost'].mean()
    user_bmi_group = pd.cut([user_data['bmi']], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])[0]
    colors_bmi = ['#FF6B6B' if group == user_bmi_group else '#F38181' for group in bmi_avg.index]
    axes[0, 2].bar(bmi_avg.index, bmi_avg.values, color=colors_bmi)
    axes[0, 2].axhline(y=prediction_usd, color='red', linestyle='--', linewidth=2, label='Your Cost')
    axes[0, 2].set_ylabel('Average Cost (USD)', fontsize=10)
    axes[0, 2].set_title('Cost by BMI Category', fontsize=12, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].tick_params(axis='x', rotation=15)
    axes[0, 2].grid(axis='y', alpha=0.3)
    smoker_avg = database.groupby('smoker')['predicted_cost'].mean()
    colors_smoker = ['#FF6B6B' if s == user_data['smoker'].lower() else '#AAF683' for s in smoker_avg.index]
    axes[1, 0].bar(smoker_avg.index, smoker_avg.values, color=colors_smoker)
    axes[1, 0].axhline(y=prediction_usd, color='red', linestyle='--', linewidth=2, label='Your Cost')
    axes[1, 0].set_ylabel('Average Cost (USD)', fontsize=10)
    axes[1, 0].set_title('Cost by Smoking Status', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    children_avg = database.groupby('children')['predicted_cost'].mean()
    colors_children = ['#FF6B6B' if c == user_data['children'] else '#FFEAA7' for c in children_avg.index]
    axes[1, 1].bar(children_avg.index, children_avg.values, color=colors_children)
    axes[1, 1].axhline(y=prediction_usd, color='red', linestyle='--', linewidth=2, label='Your Cost')
    axes[1, 1].set_xlabel('Number of Children', fontsize=10)
    axes[1, 1].set_ylabel('Average Cost (USD)', fontsize=10)
    axes[1, 1].set_title('Cost by Number of Children', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    region_avg = database.groupby('region')['predicted_cost'].mean()
    colors_region = ['#FF6B6B' if r == user_data['region'].lower() else '#DFE6E9' for r in region_avg.index]
    axes[1, 2].bar(region_avg.index, region_avg.values, color=colors_region)
    axes[1, 2].axhline(y=prediction_usd, color='red', linestyle='--', linewidth=2, label='Your Cost')
    axes[1, 2].set_ylabel('Average Cost (USD)', fontsize=10)
    axes[1, 2].set_title('Cost by Region', fontsize=12, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].tick_params(axis='x', rotation=15)
    axes[1, 2].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig
def generate_recommendations(user_data, prediction_usd, database):
    recommendations = []
    avg_cost = database['predicted_cost'].mean()
    cost_diff_pct = ((prediction_usd - avg_cost) / avg_cost) * 100
    if user_data['smoker'].lower() == 'yes':
        non_smoker_similar = database[(database['smoker'] == 'no') & 
                                       (database['age'].between(user_data['age']-5, user_data['age']+5)) &
                                       (database['bmi'].between(user_data['bmi']-2, user_data['bmi']+2))]['predicted_cost'].mean()
        potential_savings = prediction_usd - non_smoker_similar
        recommendations.append({
            'category': 'Smoking Cessation',
            'priority': 'HIGH',
            'impact': f'Potential savings: ${potential_savings:,.0f}/year (₹{potential_savings*USD_TO_INR:,.0f})',
            'action': 'Quit smoking to reduce insurance costs by 50-70%. Join smoking cessation programs, use nicotine replacement therapy, or consult a healthcare provider.',
            'timeframe': '6-12 months to see premium reductions'
        })
    if user_data['bmi'] > 30:
        normal_bmi_similar = database[(database['bmi'].between(18.5, 25)) & 
                                       (database['age'].between(user_data['age']-5, user_data['age']+5)) &
                                       (database['smoker'] == user_data['smoker'].lower())]['predicted_cost'].mean()
        potential_savings = prediction_usd - normal_bmi_similar
        target_bmi = 24.9
        current_weight_kg = user_data['bmi'] * (1.7 ** 2)
        target_weight_kg = target_bmi * (1.7 ** 2)
        weight_loss_needed = current_weight_kg - target_weight_kg
        recommendations.append({
            'category': 'Weight Management',
            'priority': 'HIGH',
            'impact': f'Potential savings: ${potential_savings:,.0f}/year (₹{potential_savings*USD_TO_INR:,.0f})',
            'action': f'Reduce BMI to normal range (18.5-24.9) by losing approximately {weight_loss_needed:.1f} kg. Consult a nutritionist, exercise 150 minutes/week, and maintain a balanced diet.',
            'timeframe': '12-18 months for sustainable weight loss'
        })
    elif user_data['bmi'] > 25:
        recommendations.append({
            'category': 'Weight Optimization',
            'priority': 'MEDIUM',
            'impact': 'Prevent future cost increases',
            'action': 'Maintain healthy weight through regular exercise (30 min daily) and balanced nutrition to prevent moving into obese category.',
            'timeframe': 'Ongoing maintenance'
        })
    if user_data['age'] < 40 and user_data['smoker'].lower() == 'no' and user_data['bmi'] < 25:
        recommendations.append({
            'category': 'Preventive Health',
            'priority': 'MEDIUM',
            'impact': 'Long-term cost stability',
            'action': 'Maintain current healthy lifestyle: annual checkups, healthy diet, regular exercise, stress management, and adequate sleep.',
            'timeframe': 'Ongoing'
        })
    similar_profile = database[(database['age'].between(user_data['age']-5, user_data['age']+5)) &
                                (database['bmi'].between(user_data['bmi']-3, user_data['bmi']+3)) &
                                (database['smoker'] == user_data['smoker'].lower())]
    if len(similar_profile) > 0:
        percentile = (similar_profile['predicted_cost'] < prediction_usd).mean() * 100
        if percentile > 75:
            recommendations.append({
                'category': 'Cost Optimization',
                'priority': 'MEDIUM',
                'impact': f'You are in the top 25% cost bracket for similar profiles',
                'action': 'Compare insurance providers, consider high-deductible health plans with HSA, review coverage needs, and look for employer wellness program discounts.',
                'timeframe': 'Next policy renewal'
            })
    recommendations.append({
        'category': 'General Wellness',
        'priority': 'LOW',
        'impact': 'Overall health improvement',
        'action': 'Regular health screenings, maintain healthy habits, manage stress, get 7-8 hours sleep, stay hydrated, and build strong social connections.',
        'timeframe': 'Ongoing lifestyle'
    })
    if cost_diff_pct > 50:
        recommendations.append({
            'category': 'Immediate Action Required',
            'priority': 'CRITICAL',
            'impact': f'Your cost is {cost_diff_pct:.1f}% higher than average',
            'action': 'Focus on high-priority recommendations immediately. Consider consulting with a health coach or financial advisor specialized in healthcare costs.',
            'timeframe': 'Start within 1 month'
        })
    return recommendations
def create_pdf_report(user_data, prediction_usd, prediction_inr, database, recommendations, chart_fig):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#2C3E50'), spaceAfter=12, alignment=TA_CENTER)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#34495E'), spaceAfter=10, spaceBefore=12)
    normal_style = styles['Normal']
    story.append(Paragraph("Health Insurance Cost Analysis Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Personal Information", heading_style))
    personal_data = [
        ['Field', 'Value'],
        ['Age', str(user_data['age'])],
        ['Gender', user_data['sex'].capitalize()],
        ['BMI', f"{user_data['bmi']:.1f}"],
        ['Children', str(user_data['children'])],
        ['Smoker', user_data['smoker'].capitalize()],
        ['Region', user_data['region'].capitalize()]
    ]
    personal_table = Table(personal_data, colWidths=[2.5*inch, 3*inch])
    personal_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(personal_table)
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Cost Prediction", heading_style))
    avg_cost = database['predicted_cost'].mean()
    cost_diff = prediction_usd - avg_cost
    cost_diff_pct = (cost_diff / avg_cost) * 100
    prediction_data = [
        ['Metric', 'Amount'],
        ['Annual Cost (USD)', f"${prediction_usd:,.2f}"],
        ['Annual Cost (INR)', f"₹{prediction_inr:,.2f}"],
        ['Monthly Cost (USD)', f"${prediction_usd/12:,.2f}"],
        ['Monthly Cost (INR)', f"₹{prediction_inr/12:,.2f}"],
        ['Database Average', f"${avg_cost:,.2f}"],
        ['Difference from Average', f"${cost_diff:,.2f} ({cost_diff_pct:+.1f}%)"]
    ]
    prediction_table = Table(prediction_data, colWidths=[2.5*inch, 3*inch])
    prediction_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27AE60')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(prediction_table)
    story.append(PageBreak())
    story.append(Paragraph("Comparative Analysis Charts", heading_style))
    img_buffer = BytesIO()
    chart_fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img = Image(img_buffer, width=7*inch, height=4.7*inch)
    story.append(img)
    story.append(PageBreak())
    story.append(Paragraph("Personalized Recommendations", heading_style))
    for i, rec in enumerate(recommendations, 1):
        priority_color = {'CRITICAL': colors.red, 'HIGH': colors.orange, 'MEDIUM': colors.yellow, 'LOW': colors.lightblue}
        rec_title = f"{i}. {rec['category']} (Priority: {rec['priority']})"
        story.append(Paragraph(rec_title, ParagraphStyle('RecTitle', parent=styles['Heading3'], fontSize=12, textColor=priority_color.get(rec['priority'], colors.black))))
        story.append(Paragraph(f"<b>Impact:</b> {rec['impact']}", normal_style))
        story.append(Paragraph(f"<b>Action:</b> {rec['action']}", normal_style))
        story.append(Paragraph(f"<b>Timeframe:</b> {rec['timeframe']}", normal_style))
        story.append(Spacer(1, 0.15*inch))
    story.append(PageBreak())
    story.append(Paragraph("Statistical Comparison", heading_style))
    age_group = pd.cut([user_data['age']], bins=[0, 30, 40, 50, 100], labels=['18-30', '31-40', '41-50', '51+'])[0]
    bmi_category = pd.cut([user_data['bmi']], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])[0]
    similar_profile = database[(database['age'].between(user_data['age']-5, user_data['age']+5)) &
                                (database['smoker'] == user_data['smoker'].lower())]
    percentile = (similar_profile['predicted_cost'] < prediction_usd).mean() * 100
    stats_data = [
        ['Category', 'Your Value', 'Database Average'],
        ['Age Group', str(age_group), 'Various'],
        ['BMI Category', str(bmi_category), 'Overweight (avg)'],
        ['Smoking Status', user_data['smoker'], '20% smokers'],
        ['Cost Percentile', f"{percentile:.1f}th", '50th (median)'],
        ['Annual Cost', f"${prediction_usd:,.0f}", f"${avg_cost:,.0f}"]
    ]
    stats_table = Table(stats_data, colWidths=[2*inch, 2*inch, 2*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9B59B6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Disclaimer", ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=9, textColor=colors.grey)))
    story.append(Paragraph("This report is generated for informational purposes only and should not be considered as medical or financial advice. Actual insurance costs may vary based on provider, coverage options, and other factors. Please consult with insurance professionals and healthcare providers for personalized recommendations.", normal_style))
    doc.build(story)
    buffer.seek(0)
    return buffer
st.markdown("# 🏥 Health Insurance Cost Predictor")
st.markdown("Enter your details to predict your annual medical insurance costs and receive personalized recommendations")
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
        user_data = {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region}
        st.success("### 💰 Prediction Results")
        col_result1, col_result2, col_result3 = st.columns(3)
        avg_cost = database['predicted_cost'].mean()
        cost_diff = prediction_usd - avg_cost
        cost_diff_pct = (cost_diff / avg_cost) * 100
        with col_result1:
            st.metric("Annual Cost", f"₹{prediction_inr:,.2f}", delta=f"{cost_diff_pct:+.1f}% vs avg")
            st.caption(f"${prediction_usd:,.2f} USD")
        with col_result2:
            st.metric("Monthly Cost", f"₹{prediction_inr/12:,.2f}")
            st.caption(f"${prediction_usd/12:,.2f} USD")
        with col_result3:
            st.metric("Database Average", f"₹{avg_cost*USD_TO_INR:,.2f}")
            st.caption(f"${avg_cost:,.2f} USD")
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
        st.markdown("### 📊 Comparative Analysis")
        fig = create_comparison_charts(user_data, prediction_usd, database)
        st.pyplot(fig)
        plt.close()
        st.divider()
        st.markdown("### 💡 Personalized Recommendations")
        recommendations = generate_recommendations(user_data, prediction_usd, database)
        for rec in recommendations:
            priority_colors = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
            with st.expander(f"{priority_colors.get(rec['priority'], '⚪')} {rec['category']} - {rec['priority']} Priority"):
                st.markdown(f"**Impact:** {rec['impact']}")
                st.markdown(f"**Recommended Action:** {rec['action']}")
                st.markdown(f"**Timeframe:** {rec['timeframe']}")
        st.divider()
        st.markdown("### 📥 Download Complete Report")
        pdf_buffer = create_pdf_report(user_data, prediction_usd, prediction_inr, database, recommendations, fig)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer,
            file_name=f"insurance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        with st.expander("📊 Additional Cost Factors"):
            factors = []
            if smoker == "Yes":
                factors.append("• Smoking significantly increases insurance costs (typically 2-3x higher)")
            if bmi > 30:
                factors.append("• BMI over 30 may increase premiums by 20-50%")
            if age > 50:
                factors.append("• Age over 50 typically increases costs due to higher health risks")
            if children > 2:
                factors.append("• Multiple dependents may affect family plan costs")
            if cost_diff_pct > 25:
                factors.append(f"• Your cost is {cost_diff_pct:.1f}% above average - review recommendations")
            if factors:
                for factor in factors:
                    st.write(factor)
            else:
                st.write("✅ No significant risk factors identified - maintain healthy lifestyle!")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.exception(e)
