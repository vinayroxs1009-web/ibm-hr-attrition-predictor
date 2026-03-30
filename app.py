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

# ── Full Dark Theme CSS ───────────────────────────────────────
st.markdown("""
<style>
    .stApp, .main, section.main, [data-testid="stAppViewContainer"] {
        background-color: #0F1117 !important;
        color: #EAEAEA !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1A1D2E !important;
    }
    [data-testid="stSidebar"] * { color: #EAEAEA !important; }
    h1,h2,h3,h4,h5,h6,p,label,span,div { color: #EAEAEA !important; }
    .stSelectbox label, .stSlider label {
        color: #AAAAAA !important; font-size: 13px !important;
    }
    .stSelectbox > div > div {
        background-color: #1A1D2E !important;
        color: #EAEAEA !important;
        border: 1px solid #2A2D3E !important;
        border-radius: 8px !important;
    }
    hr { border-color: #2A2D3E !important; }
    .stAlert {
        background-color: #1A1D2E !important;
        border: 1px solid #2A2D3E !important;
        color: #EAEAEA !important;
        border-radius: 10px !important;
    }
    .metric-card {
        background: #1A1D2E;
        border-radius: 14px;
        padding: 22px 24px;
        border-left: 4px solid #E63946;
        margin: 8px 0;
    }
    .risk-high { color: #E63946; font-size: 32px; font-weight: 800; }
    .risk-med  { color: #FF9800; font-size: 32px; font-weight: 800; }
    .risk-low  { color: #4CAF50; font-size: 32px; font-weight: 800; }
    .metric-label { color: #888; font-size: 13px; margin: 0 0 4px 0; }
    .metric-sub   { color: #888; font-size: 12px; margin: 6px 0 0 0; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark defaults ──────────────────────────────────
BG   = '#0F1117'
CARD = '#1A1D2E'
TEXT = '#EAEAEA'
GRID = '#2A2D3E'
RED  = '#E63946'
BLUE = '#2196F3'

plt.rcParams.update({
    'figure.facecolor': BG,  'axes.facecolor': CARD,
    'axes.edgecolor':  GRID, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT,     'ytick.color': TEXT,
    'text.color': TEXT,      'grid.color': GRID,
    'font.family': 'DejaVu Sans', 'font.size': 11,
})

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('xgb_model.pkl')

model = load_model()

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#EAEAEA;font-size:32px;font-weight:800;margin-bottom:4px'>
🏢 IBM HR Attrition Predictor</h1>
<p style='color:#888;font-size:15px;margin-top:0'>
Identify at-risk employees before they leave — XGBoost + SHAP Explainability</p>
""", unsafe_allow_html=True)
st.divider()

# ── Sidebar inputs ────────────────────────────────────────────
st.sidebar.markdown("<h2 style='color:#EAEAEA;font-size:20px;font-weight:700'>👤 Employee Profile</h2>", unsafe_allow_html=True)

age                  = st.sidebar.slider("Age", 18, 60, 32)
monthly_income       = st.sidebar.slider("Monthly Income ($)", 1000, 20000, 5000, 500)
job_satisfaction     = st.sidebar.selectbox("Job Satisfaction (1=Low → 4=High)", [1,2,3,4], index=2)
overtime             = st.sidebar.selectbox("Works OverTime?", ["Yes","No"])
work_life_balance    = st.sidebar.selectbox("Work-Life Balance (1=Bad → 4=Best)", [1,2,3,4], index=2)
distance_from_home   = st.sidebar.slider("Distance from Home (km)", 1, 30, 8)
years_at_company     = st.sidebar.slider("Years at Company", 0, 40, 5)
num_companies_worked = st.sidebar.slider("No. of Companies Worked", 0, 10, 2)
job_level            = st.sidebar.selectbox("Job Level (1=Junior → 5=Senior)", [1,2,3,4,5], index=1)
marital_status       = st.sidebar.selectbox("Marital Status", ["Single","Married","Divorced"])
environment_sat      = st.sidebar.selectbox("Environment Satisfaction (1-4)", [1,2,3,4], index=2)
years_since_promo    = st.sidebar.slider("Years Since Last Promotion", 0, 15, 2)
total_working_years  = st.sidebar.slider("Total Working Years", 0, 40, 8)
training_times       = st.sidebar.slider("Training Times Last Year", 0, 6, 2)
performance_rating   = st.sidebar.selectbox("Performance Rating (3=Meets / 4=Exceeds)", [3,4])
stock_option_level   = st.sidebar.selectbox("Stock Option Level (0-3)", [0,1,2,3])

# ── Build input row ───────────────────────────────────────────
marital_map  = {"Single": 2, "Married": 1, "Divorced": 0}
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
    'YearsSinceLastPromotion': years_since_promo,
    'YearsWithCurrManager': min(years_at_company, 4)
}])

# ── Predict ───────────────────────────────────────────────────
proba        = model.predict_proba(input_data)[0][1]
risk_pct     = round(proba * 100, 1)
replace_cost = int(monthly_income * 12 * 0.5)

if proba >= 0.65:
    risk_label = "🔴 HIGH RISK";   risk_class = "risk-high"
    advice     = "⚠️ Immediate intervention needed. Schedule 1:1, review compensation and workload."
    bar_color  = RED
elif proba >= 0.40:
    risk_label = "🟡 MEDIUM RISK"; risk_class = "risk-med"
    advice     = "👀 Monitor closely. Consider mentorship, role enrichment, or a salary review."
    bar_color  = '#FF9800'
else:
    risk_label = "🟢 LOW RISK";    risk_class = "risk-low"
    advice     = "✅ Employee appears stable. Continue regular check-ins and recognition."
    bar_color  = '#4CAF50'

# ── Metric cards ──────────────────────────────────────────────
c1, c2, c3 = st.columns([1,1,2])
with c1:
    st.markdown(f"""<div class='metric-card'>
        <p class='metric-label'>Attrition Risk Score</p>
        <p class='{risk_class}'>{risk_pct}%</p>
        <p style='color:#EAEAEA;font-size:16px;font-weight:600;margin:4px 0'>{risk_label}</p>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='metric-card' style='border-left-color:#FF9800'>
        <p class='metric-label'>Est. Replacement Cost</p>
        <p style='color:#FF9800;font-size:28px;font-weight:800'>${replace_cost:,}</p>
        <p class='metric-sub'>~6 months salary if employee leaves</p>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='metric-card' style='border-left-color:#2196F3;padding:28px 24px'>
        <p class='metric-label'>HR Recommendation</p>
        <p style='color:#EAEAEA;font-size:15px;font-weight:500;margin:6px 0'>{advice}</p>
    </div>""", unsafe_allow_html=True)

# ── Risk gauge bar ────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
fig_g, ax_g = plt.subplots(figsize=(10, 1.3), facecolor=BG)
ax_g.set_facecolor(BG)
ax_g.barh(0, 100, color=GRID, height=0.4)
ax_g.barh(0, risk_pct, color=bar_color, height=0.4)
ax_g.set_xlim(0, 100); ax_g.set_yticks([])
ax_g.axvline(40, color='#FF9800', linestyle='--', linewidth=1.2, alpha=0.7)
ax_g.axvline(65, color=RED,       linestyle='--', linewidth=1.2, alpha=0.7)
ax_g.text(20, 0.35, 'LOW',    color='#4CAF50', fontsize=9, ha='center', fontweight='bold')
ax_g.text(52, 0.35, 'MEDIUM', color='#FF9800', fontsize=9, ha='center', fontweight='bold')
ax_g.text(82, 0.35, 'HIGH',   color=RED,       fontsize=9, ha='center', fontweight='bold')
ax_g.text(risk_pct, -0.35, f'{risk_pct}%', color=bar_color,
          fontsize=12, ha='center', fontweight='bold')
ax_g.set_xlabel("Attrition Risk %", color=TEXT, fontsize=11)
ax_g.spines[:].set_visible(False)
plt.tight_layout()
st.pyplot(fig_g, use_container_width=True)
plt.close()

st.divider()

# ── SHAP dark bar chart ───────────────────────────────────────
st.markdown("""
<h3 style='color:#EAEAEA;font-size:20px;font-weight:700'>🔍 Why Is This Employee At Risk?</h3>
<p style='color:#888;font-size:14px'>Each bar shows how much a feature pushes risk
<span style='color:#E63946;font-weight:700'>UP ↑ (red)</span> or
<span style='color:#2196F3;font-weight:700'>DOWN ↓ (blue)</span></p>
""", unsafe_allow_html=True)

explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(input_data)

shap_series = pd.Series(shap_vals[0], index=input_data.columns)
top_shap    = shap_series.reindex(shap_series.abs().sort_values(ascending=False).index).head(10)
top_shap    = top_shap.sort_values()
bar_colors  = [RED if v > 0 else BLUE for v in top_shap.values]

fig_s, ax_s = plt.subplots(figsize=(10, 6), facecolor=BG)
ax_s.set_facecolor(CARD)
bars = ax_s.barh(top_shap.index, top_shap.values,
                 color=bar_colors, edgecolor=BG, height=0.6)
for bar, val in zip(bars, top_shap.values):
    ax_s.text(
        val + (0.04 if val >= 0 else -0.04),
        bar.get_y() + bar.get_height()/2,
        f'{val:+.2f}', va='center',
        ha='left' if val >= 0 else 'right',
        color=TEXT, fontsize=10, fontweight='bold'
    )
ax_s.axvline(0, color=TEXT, linewidth=0.8, alpha=0.4)
ax_s.set_xlabel("SHAP Value  (impact on attrition risk)", color=TEXT, fontsize=11)
ax_s.set_title("Top 10 Features Driving This Prediction",
               color=TEXT, fontsize=13, fontweight='bold', pad=14)
ax_s.tick_params(colors=TEXT, labelsize=10)
ax_s.spines['top'].set_visible(False)
ax_s.spines['right'].set_visible(False)
ax_s.spines['left'].set_color(GRID)
ax_s.spines['bottom'].set_color(GRID)
ax_s.grid(axis='x', alpha=0.15, color=GRID)
plt.tight_layout()
st.pyplot(fig_s, use_container_width=True)
plt.close()

st.divider()

# ── Team risk table ───────────────────────────────────────────
st.markdown("""
<h3 style='color:#EAEAEA;font-size:20px;font-weight:700'>📊 Simulated Team Risk Overview</h3>
<p style='color:#888;font-size:14px'>10 employees with varied profiles — who needs attention?</p>
""", unsafe_allow_html=True)

sample = pd.DataFrame({
    'Employee':        [f'EMP-{i:03d}' for i in range(1,11)],
    'Age':             [24,38,42,29,55,31,47,26,33,50],
    'Monthly Income':  ['$2,500','$8,000','$12,000','$3,200','$15,000',
                        '$4,000','$11,000','$2,800','$6,000','$18,000'],
    'OverTime':        ['Yes','No','No','Yes','No','Yes','No','Yes','No','No'],
    'Job Satisfaction':['Low','High','High','Low','High','Low','High','Low','Medium','High'],
    'Risk Level':      ['🔴 High','🟢 Low','🟢 Low','🟡 Medium','🟢 Low',
                        '🔴 High','🟢 Low','🔴 High','🟢 Low','🟢 Low']
})
st.dataframe(sample, use_container_width=True, hide_index=True)

st.divider()
st.markdown("<p style='color:#333;font-size:12px;text-align:center'>Built with XGBoost + SHAP | IBM HR Attrition Dataset | Data Science Portfolio Project</p>",
            unsafe_allow_html=True)
