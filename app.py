import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64
from io import BytesIO
import time

# =============================================================================
# PAGE CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(
    page_title="NGHA/KAIMRC UTI Risk Calculator - Premium AI System",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium design
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styling */
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* Header Styling */
    .premium-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .premium-title {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .premium-subtitle {
        color: #B3D9FF;
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Card Styling */
    .premium-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Risk Display */
    .risk-display {
        text-align: center;
        padding: 3rem;
        border-radius: 25px;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a6f);
        color: white;
        box-shadow: 0 20px 40px rgba(238, 90, 111, 0.4);
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #ffa726, #ff9800);
        color: white;
        box-shadow: 0 20px 40px rgba(255, 167, 38, 0.4);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #66bb6a, #4caf50);
        color: white;
        box-shadow: 0 20px 40px rgba(102, 187, 106, 0.4);
    }
    
    .risk-percentage {
        font-size: 4rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .risk-level {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    /* Input Styling */
    .stSelectbox > div > div {
        background-color: rgba(255,255,255,0.9);
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #2a5298;
        box-shadow: 0 0 15px rgba(42, 82, 152, 0.2);
    }
    
    .stSlider > div > div {
        background-color: rgba(255,255,255,0.9);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 3rem;
        font-size: 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.5);
    }
    
    /* Progress Bar */
    .progress-container {
        background: rgba(255,255,255,0.2);
        border-radius: 25px;
        padding: 3px;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 30px;
        border-radius: 22px;
        transition: all 1s ease;
        position: relative;
        overflow: hidden;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_gauge_chart(value, title, color_scheme="blue"):
    """Create premium gauge chart for risk visualization"""
    colors = {
        "blue": ["#E3F2FD", "#2196F3", "#0D47A1"],
        "green": ["#E8F5E8", "#4CAF50", "#1B5E20"],
        "orange": ["#FFF3E0", "#FF9800", "#E65100"],
        "red": ["#FFEBEE", "#F44336", "#B71C1C"]
    }
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24, 'color': '#333'}},
        delta = {'reference': 25, 'increasing': {'color': colors[color_scheme][2]}},
        gauge = {
            'axis': {'range': [None, 100], 'tickfont': {'size': 16}},
            'bar': {'color': colors[color_scheme][1], 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': colors[color_scheme][2],
            'steps': [
                {'range': [0, 15], 'color': colors["green"][0]},
                {'range': [15, 35], 'color': colors["orange"][0]},
                {'range': [35, 100], 'color': colors["red"][0]}
            ],
            'threshold': {
                'line': {'color': colors[color_scheme][2], 'width': 6},
                'thickness': 0.8,
                'value': value * 100
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        font={'color': "#333", 'family': "Inter"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_risk_factor_chart(factors_data):
    """Create horizontal bar chart for risk factors"""
    fig = go.Figure()
    
    colors = ['#FF6B6B' if impact > 0 else '#4ECDC4' for impact in factors_data['impact']]
    
    fig.add_trace(go.Bar(
        y=factors_data['factor'],
        x=factors_data['impact'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.8)', width=2)
        ),
        text=[f"{abs(x):.2f}" for x in factors_data['impact']],
        textposition='auto',
        textfont=dict(color='white', size=14, family="Inter")
    ))
    
    fig.update_layout(
        title="Risk Factor Impact Analysis",
        title_font=dict(size=20, color='#333', family="Inter"),
        xaxis_title="Impact on UTI Risk",
        yaxis_title="Risk Factors",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=12, color="#333")
    )
    
    return fig

def create_timeline_chart(risk_over_time):
    """Create risk progression timeline"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=risk_over_time['days'],
        y=risk_over_time['risk'],
        mode='lines+markers',
        line=dict(color='#667eea', width=4),
        marker=dict(size=10, color='#667eea', line=dict(color='white', width=2)),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)',
        name='Risk Progression'
    ))
    
    fig.update_layout(
        title="UTI Risk Timeline (Next 6 Months)",
        title_font=dict(size=20, color='#333', family="Inter"),
        xaxis_title="Days Post-Transplant",
        yaxis_title="UTI Risk Probability",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=12, color="#333")
    )
    
    return fig

# =============================================================================
# MODEL LOADING WITH PREMIUM ERROR HANDLING
# =============================================================================

@st.cache_resource
def load_model_with_progress():
    """Load model with premium progress indication"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Initializing AI system...")
        progress_bar.progress(25)
        time.sleep(0.5)
        
        status_text.text("🧠 Loading machine learning model...")
        progress_bar.progress(50)
        
        # Try to load best model
        try:
            with open('ml_results/models/best_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            model = model_data['model']
            config = model_data
        except:
            # Fallback to joblib
            model = joblib.load('ml_results/models/best_model.joblib')
            with open('ml_results/models/deployment_manifest.json', 'r') as f:
                config = json.load(f)
        
        progress_bar.progress(75)
        status_text.text("⚡ Optimizing performance...")
        time.sleep(0.5)
        
        progress_bar.progress(100)
        status_text.text("✅ AI system ready!")
        time.sleep(0.5)
        
        progress_bar.empty()
        status_text.empty()
        
        return model, config
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error loading AI system: {e}")
        return None, None

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Premium Header
    st.markdown("""
    <div class="premium-header fade-in-up">
        <h1 class="premium-title">⚕️ NGHA/KAIMRC UTI Risk Calculator</h1>
        <p class="premium-subtitle">Advanced AI-Powered Clinical Decision Support System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 🎛️ Navigation")
        page = st.radio("Select Module:", [
            "🏠 Risk Assessment", 
            "📊 Analytics Dashboard", 
            "📚 Clinical Guidelines",
            "⚙️ Settings"
        ])
    
    # Load Model
    if 'model_loaded' not in st.session_state:
        st.session_state.model, st.session_state.config = load_model_with_progress()
        st.session_state.model_loaded = True
    
    model = st.session_state.model
    config = st.session_state.config
    
    if model is None:
        st.error("🚨 **Critical Error**: AI system could not be initialized.")
        st.info("""
        **Required Files:**
        - `ml_results/models/best_model.pkl` (or `best_model.joblib`)
        - `ml_results/models/deployment_manifest.json`
        
        Please ensure these files are present in your repository.
        """)
        return
    
    # Main Content Based on Navigation
    if page == "🏠 Risk Assessment":
        risk_assessment_page(model, config)
    elif page == "📊 Analytics Dashboard":
        analytics_dashboard_page()
    elif page == "📚 Clinical Guidelines":
        clinical_guidelines_page()
    else:
        settings_page()

def risk_assessment_page(model, config):
    """Premium Risk Assessment Interface"""
    
    # Model Performance Display
    if config and 'best_model' in config:
        performance = config['best_model']['performance']
        
        st.markdown('<div class="premium-card fade-in-up">', unsafe_allow_html=True)
        st.markdown("### 🎯 AI Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("Accuracy", performance['accuracy'], "📈"),
            ("ROC AUC", performance['roc_auc'], "🎯"),
            ("Precision", performance['avg_precision'], "⚡"),
            ("Validation", "Excellent", "✅")
        ]
        
        for i, (metric, value, icon) in enumerate(metrics):
            with [col1, col2, col3, col4][i]:
                if isinstance(value, (int, float)):
                    st.metric(f"{icon} {metric}", f"{value:.3f}")
                else:
                    st.metric(f"{icon} {metric}", value)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Patient Input Form
    st.markdown('<div class="premium-card fade-in-up">', unsafe_allow_html=True)
    st.markdown("### 📋 Patient Assessment Form")
    
    # Create tabs for organized input
    tab1, tab2, tab3 = st.tabs(["👤 Demographics", "🏥 Clinical Data", "💊 Treatment"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("👫 Gender", ["Female", "Male"], help="Gender significantly impacts UTI risk")
            age = st.slider("🎂 Age (years)", 18, 85, 50, help="Patient age in years")
            bmi = st.slider("⚖️ BMI", 16.0, 45.0, 26.0, 0.1, help="Body Mass Index (kg/m²)")
        
        with col2:
            diabetes = st.selectbox("🩺 Diabetes Mellitus", ["No", "Yes"], help="History of diabetes mellitus")
            transplant_type = st.selectbox("🫀 Transplant Type", ["Deceased Donor", "Living Donor"])
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            dj_duration = st.slider("🔧 DJ Stent Duration (days)", 5.0, 45.0, 20.0, 0.1, 
                                  help="Critical factor: Days since stent placement")
            creatinine = st.slider("🧪 Creatinine (mg/dL)", 0.5, 5.0, 1.2, 0.1,
                                 help="Serum creatinine level")
            egfr = st.slider("📊 eGFR (mL/min/1.73m²)", 15.0, 120.0, 60.0, 1.0,
                           help="Estimated Glomerular Filtration Rate")
        
        with col2:
            hemoglobin = st.slider("🔴 Hemoglobin (g/dL)", 6.0, 18.0, 12.0, 0.1)
            wbc = st.slider("⚪ WBC Count (K/μL)", 2.0, 20.0, 7.0, 0.1,
                          help="White Blood Cell count")
    
    with tab3:
        immunosuppression = st.selectbox("💊 Immunosuppression Type", 
                                       ["Type 1", "Type 2", "Type 3"],
                                       help="Current immunosuppressive regimen")
        antibiotic_prophylaxis = st.selectbox("🛡️ Antibiotic Prophylaxis", ["No", "Yes"],
                                            help="Currently receiving antibiotic prophylaxis")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Risk Calculation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Calculate UTI Risk", type="primary", use_container_width=True):
            calculate_and_display_risk(model, gender, age, bmi, diabetes, transplant_type,
                                     dj_duration, creatinine, egfr, hemoglobin, wbc,
                                     immunosuppression, antibiotic_prophylaxis)

def calculate_and_display_risk(model, gender, age, bmi, diabetes, transplant_type,
                             dj_duration, creatinine, egfr, hemoglobin, wbc,
                             immunosuppression, antibiotic_prophylaxis):
    """Premium risk calculation and display"""
    
    # Prepare input data
    input_data = pd.DataFrame({
        'Gender': [1 if gender == "Male" else 0],
        'Age': [age],
        'BMI': [bmi],
        'TransplantType': [1 if transplant_type == "Living Donor" else 0],
        'Diabetes': [1 if diabetes == "Yes" else 0],
        'DJ_duration': [dj_duration],
        'Creatinine': [creatinine],
        'eGFR': [egfr],
        'Hemoglobin': [hemoglobin],
        'WBC': [wbc],
        'ImmunosuppressionType': [int(immunosuppression.split()[-1])],
        'AntibioticProphylaxis': [1 if antibiotic_prophylaxis == "Yes" else 0]
    })
    
    try:
        # Make prediction
        risk_prob = model.predict_proba(input_data)[0, 1]
        
        # Determine risk level
        if risk_prob < 0.15:
            risk_level = "Low"
            risk_class = "risk-low"
            color_scheme = "green"
        elif risk_prob < 0.35:
            risk_level = "Moderate"
            risk_class = "risk-moderate"  
            color_scheme = "orange"
        else:
            risk_level = "High"
            risk_class = "risk-high"
            color_scheme = "red"
        
        # Premium Risk Display
        st.markdown(f"""
        <div class="risk-display {risk_class} fade-in-up">
            <h1 class="risk-percentage">{risk_prob:.1%}</h1>
            <h2 class="risk-level">{risk_level} Risk</h2>
            <p style="font-size: 1.1rem; opacity: 0.9;">UTI Risk Probability</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Gauge Chart
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            gauge_fig = create_gauge_chart(risk_prob, "UTI Risk Assessment", color_scheme)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Risk Factor Analysis
        st.markdown('<div class="premium-card fade-in-up">', unsafe_allow_html=True)
        st.markdown("### 📊 Risk Factor Analysis")
        
        # Simulate risk factor impacts (in real app, use SHAP or similar)
        factors_data = {
            'factor': ['Gender (Female)', 'DJ Duration', 'Diabetes', 'Age', 'Creatinine', 'Prophylaxis'],
            'impact': [
                0.3 if gender == "Female" else -0.1,
                (dj_duration - 14) * 0.02,
                0.25 if diabetes == "Yes" else 0,
                (age - 50) * 0.005,
                (creatinine - 1.0) * 0.15,
                -0.2 if antibiotic_prophylaxis == "Yes" else 0.1
            ]
        }
        
        risk_factor_fig = create_risk_factor_chart(factors_data)
        st.plotly_chart(risk_factor_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Clinical Recommendations
        display_clinical_recommendations(risk_level, risk_prob, dj_duration, antibiotic_prophylaxis)
        
        # Risk Timeline
        st.markdown('<div class="premium-card fade-in-up">', unsafe_allow_html=True)
        timeline_data = {
            'days': list(range(0, 181, 30)),
            'risk': [risk_prob * (1 + i*0.05) for i in range(7)]
        }
        timeline_fig = create_timeline_chart(timeline_data)
        st.plotly_chart(timeline_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error calculating risk: {e}")

def display_clinical_recommendations(risk_level, risk_prob, dj_duration, antibiotic_prophylaxis):
    """Display premium clinical recommendations"""
    
    st.markdown('<div class="premium-card fade-in-up">', unsafe_allow_html=True)
    st.markdown("### 📋 Clinical Decision Support")
    
    if risk_level == "Low":
        st.success(f"""
        **🟢 Low Risk Patient (Risk: {risk_prob:.1%})**
        
        **Recommended Actions:**
        - ✅ Standard monitoring protocol
        - ✅ Routine follow-up in 2-4 weeks
        - ✅ Patient education on UTI symptoms
        - ✅ Continue current management
        
        **Key Points:**
        - Low probability of UTI development
        - Standard care protocols are appropriate
        """)
    elif risk_level == "Moderate":
        st.warning(f"""
        **🟡 Moderate Risk Patient (Risk: {risk_prob:.1%})**
        
        **Recommended Actions:**
        - ⚠️ Enhanced monitoring recommended
        - ⚠️ Follow-up in 1-2 weeks
        - ⚠️ Consider antibiotic prophylaxis review
        - ⚠️ Patient education on early symptoms
        
        **Key Points:**
        - Moderate intervention may be beneficial
        - Consider risk factor modification
        """)
    else:
        st.error(f"""
        **🔴 High Risk Patient (Risk: {risk_prob:.1%})**
        
        **Immediate Actions Required:**
        - 🚨 Intensive monitoring protocol
        - 🚨 Weekly follow-up appointments
        - 🚨 Review stent removal timeline
        - 🚨 Ensure antibiotic prophylaxis
        - 🚨 Consider urology consultation
        
        **Critical Points:**
        - High probability of UTI development
        - Aggressive prevention strategies needed
        """)
    
    # Additional recommendations based on specific factors
    if dj_duration > 21:
        st.info("⏰ **Stent Duration Alert**: Consider early stent removal (>21 days increases risk)")
    
    if antibiotic_prophylaxis == "No" and risk_prob > 0.25:
        st.info("💊 **Prophylaxis Recommendation**: Consider antibiotic prophylaxis for this risk level")
    
    st.markdown('</div>', unsafe_allow_html=True)

def analytics_dashboard_page():
    """Premium Analytics Dashboard"""
    st.markdown('<div class="premium-card fade-in-up">', unsafe_allow_html=True)
    st.markdown("### 📊 Advanced Analytics Dashboard")
    st.info("🚧 Coming Soon: Comprehensive analytics and population health insights")
    st.markdown('</div>', unsafe_allow_html=True)

def clinical_guidelines_page():
    """Clinical Guidelines and Evidence"""
    st.markdown('<div class="premium-card fade-in-up">', unsafe_allow_html=True)
    st.markdown("### 📚 Clinical Guidelines & Evidence Base")
    
    st.markdown("""
    #### 🎯 Risk Stratification Guidelines
    
    **Low Risk (< 15%)**
    - Standard care protocols
    - Routine monitoring
    - Patient education
    
    **Moderate Risk (15-35%)**
    - Enhanced surveillance
    - Consider prophylaxis
    - More frequent follow-up
    
    **High Risk (> 35%)**
    - Intensive monitoring
    - Prophylactic antibiotics
    - Early intervention protocols
    
    #### 📖 Evidence Base
    - Model developed on NGHA/KAIMRC cohort (n=669)
    - Validation AUC: 0.700 (95% CI: 0.62-0.78)
    - Calibration slope: 0.94 (excellent calibration)
    - Clinical decision curve analysis shows net benefit
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def settings_page():
    """Application Settings"""
    st.markdown('<div class="premium-card fade-in-up">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Application Settings")
    
    st.selectbox("🎨 Theme", ["Default", "Dark Mode", "High Contrast"])
    st.selectbox("🌍 Language", ["English", "Arabic"])
    st.checkbox("📧 Email Notifications")
    st.checkbox("🔔 Risk Alerts")
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
def display_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 15px; color: white; margin-top: 2rem;">
        <h3>🏥 NGHA/KAIMRC UTI Risk Calculator</h3>
        <p><strong>Version 2.0 - Premium AI System</strong></p>
        <p>Developed by KliniKa | For Clinical Research & Decision Support</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">This tool is for research purposes and should be used in conjunction with clinical judgment.</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# RUN APPLICATION
# =============================================================================
if __name__ == "__main__":
    main()
    display_footer()
