import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="IBM HR Attrition Predictor",
    page_icon="🏢",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0F1117; }
    .stApp { background-color: #0F1117; }
    .metric-card {
        background: #1A1D2E;
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #E63946;
        margin: 8px 0;
    }
    .risk-high { color: #E63946; font-size: 28px; font-weight: bold; }
    .risk-med  { color: #FF9800; font-size: 28px; font-weight: bold; }
    .risk-low  { color: #4CAF50; font-size: 28px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('xgb_model.pkl')

model = load_model()

# ── Header ────────────────────────────────────────────────────
st.title("🏢 IBM HR Attrition Predictor")
st.markdown("**Identify at-risk employees before they leave. Built with XGBoost + SHAP explainability.**")
st.divider()

# ── Sidebar Inputs ────────────────────────────────────────────
st.sidebar.header("👤 Employee Profile")
st.sidebar.markdown("Adjust the sliders to match the employee's details.")

age                     = st.sidebar.slider("Age", 18, 60, 32)
monthly_income          = st.sidebar.slider("Monthly Income ($)", 1000, 20000, 5000, 500)
job_satisfaction        = st.sidebar.selectbox("Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4], index=2)
overtime                = st.sidebar.selectbox("Works OverTime?", ["Yes", "No"])
work_life_balance       = st.sidebar.selectbox("Work-Life Balance (1=Bad, 4=Best)", [1, 2, 3, 4], index=2)
distance_from_home      = st.sidebar.slider("Distance from Home (km)", 1, 30, 8)
years_at_company        = st.sidebar.slider("Years at Company", 0, 40, 5)
num_companies_worked    = st.sidebar.slider("No. of Companies Worked", 0, 10, 2)
job_level               = st.sidebar.selectbox("Job Level (1=Junior, 5=Senior)", [1, 2, 3, 4, 5], index=1)
marital_status          = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
environment_sat         = st.sidebar.selectbox("Environment Satisfaction (1-4)", [1, 2, 3, 4], index=2)
years_since_promotion   = st.sidebar.slider("Years Since Last Promotion", 0, 15, 2)
total_working_years     = st.sidebar.slider("Total Working Years", 0, 40, 8)
training_times          = st.sidebar.slider("Training Times Last Year", 0, 6, 2)
performance_rating      = st.sidebar.selectbox("Performance Rating (3=Meets, 4=Exceeds)", [3, 4])
stock_option_level      = st.sidebar.selectbox("Stock Option Level (0-3)", [0, 1, 2, 3])

# ── Build Input Row ───────────────────────────────────────────
marital_map = {"Single": 2, "Married": 1, "Divorced": 0}
overtime_map = {"Yes": 1, "No": 0}

input_data = pd.DataFrame([{
    'Age': age, 'BusinessTravel': 1, 'Department': 1,
    'DistanceFromHome': distance_from_home, 'Education': 3,
    'EducationField': 1, 'EnvironmentSatisfaction': environment_sat,
    'Gender': 1, 'JobInvolvement': 3, 'JobLevel': job_level,
    'JobRole': 1, 'JobSatisfaction': job_satisfaction,
    'MaritalStatus': marital_map[marital_status],
    'MonthlyIncome': monthly_income,
    'NumCompaniesWorked': num_companies_worked,
    'OverTime': overtime_map[overtime],
    'PerformanceRating': performance_rating,
    'RelationshipSatisfaction': 3,
    'StockOptionLevel': stock_option_level,
    'TotalWorkingYears': total_working_years,
    'TrainingTimesLastYear': training_times,
    'WorkLifeBalance': work_life_balance,
    'YearsAtCompany': years_at_company,
    'YearsInCurrentRole': min(years_at_company, 3),
    'YearsSinceLastPromotion': years_since_promotion,
    'YearsWithCurrManager': min(years_at_company, 4)
}])

# ── Predict ───────────────────────────────────────────────────
proba = model.predict_proba(input_data)[0][1]
risk_pct = round(proba * 100, 1)

if proba >= 0.65:
    risk_label = "🔴 HIGH RISK"
    risk_class = "risk-high"
    advice = "Immediate retention intervention recommended. Schedule 1:1, review compensation, workload."
elif proba >= 0.40:
    risk_label = "🟡 MEDIUM RISK"
    risk_class = "risk-med"
    advice = "Monitor closely. Consider mentorship program or role enrichment."
else:
    risk_label = "🟢 LOW RISK"
    risk_class = "risk-low"
    advice = "Employee appears stable. Continue regular check-ins."

# ── Main Display ──────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <p style='color:#aaa; margin:0'>Attrition Risk Score</p>
        <p class='{risk_class}'>{risk_pct}%</p>
        <p style='color:white; font-size:16px'>{risk_label}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <p style='color:#aaa; margin:0'>Replacement Cost Estimate</p>
        <p style='color:#FF9800; font-size:22px; font-weight:bold'>
            ${int(monthly_income * 12 * 0.5):,}
        </p>
        <p style='color:#aaa; font-size:12px'>~6 months salary if employee leaves</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.info(f"**HR Recommendation:** {advice}")

st.divider()

# ── SHAP Waterfall ────────────────────────────────────────────
st.subheader("🔍 Why Is This Employee At Risk? (SHAP Explainability)")
st.markdown("Each bar shows how much a feature **pushes the risk up (red) or down (blue)**.")

explainer   = shap.TreeExplainer(model)
shap_vals   = explainer.shap_values(input_data)
exp = shap.Explanation(
    values=shap_vals[0],
    base_values=explainer.expected_value,
    data=input_data.iloc[0].values,
    feature_names=input_data.columns.tolist()
)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#0F1117')
shap.waterfall_plot(exp, max_display=10, show=False)
st.pyplot(fig)
plt.close()

# ── Batch Risk Table ──────────────────────────────────────────
st.divider()
st.subheader("📊 Simulated Team Risk Overview")
st.markdown("Example: 10 employees with varied profiles — who's most at risk?")

sample_data = pd.DataFrame({
    'Employee': [f'EMP-{i:03d}' for i in range(1, 11)],
    'Age': [24, 38, 42, 29, 55, 31, 47, 26, 33, 50],
    'MonthlyIncome': [2500, 8000, 12000, 3200, 15000, 4000, 11000, 2800, 6000, 18000],
    'OverTime': ['Yes','No','No','Yes','No','Yes','No','Yes','No','No'],
    'JobSatisfaction': [1, 3, 4, 2, 4, 1, 3, 2, 3, 4],
    'Risk': ['🔴 High', '🟢 Low', '🟢 Low', '🟡 Medium', '🟢 Low',
             '🔴 High', '🟢 Low', '🔴 High', '🟢 Low', '🟢 Low']
})
st.dataframe(sample_data, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("*Built with XGBoost + SHAP | IBM HR Attrition Dataset | Portfolio Project — [Your Name]*")
