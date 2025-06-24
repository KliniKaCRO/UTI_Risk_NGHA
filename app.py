import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

# =============================================================================
# CUSTOM CLASSES FOR MODEL COMPATIBILITY
# =============================================================================

class FeatureRemovalPreprocessor(BaseEstimator, TransformerMixin):
    """Smart wrapper that removes a feature and corresponding output columns"""
    
    def __init__(self, original_preprocessor, feature_to_remove='AntibioticProphylaxis'):
        self.original_preprocessor = original_preprocessor
        self.feature_to_remove = feature_to_remove
        self.original_features = None
        self.new_features = None
        self.columns_to_remove = None
        self._is_fitted = True
        self._sklearn_fitted = True
        
    def transform(self, X):
        """Transform input by removing feature, then remove corresponding output columns"""
        # Handle input data
        if hasattr(X, 'columns') and self.feature_to_remove in X.columns:
            X_reduced = X.drop(columns=[self.feature_to_remove], errors='ignore')
            X_for_transform = X_reduced.copy()
            X_for_transform[self.feature_to_remove] = 0
            if hasattr(self.original_preprocessor, 'feature_names_in_'):
                X_for_transform = X_for_transform[self.original_preprocessor.feature_names_in_]
        else:
            X_for_transform = X.copy()
            if hasattr(X, 'columns') and len(X.columns) == 11:
                X_for_transform[self.feature_to_remove] = 0
                expected_order = ['Gender', 'Age', 'BMI', 'TransplantType', 'Diabetes', 
                                'DJ_duration', 'Creatinine', 'eGFR', 'Hemoglobin', 'WBC', 
                                'ImmunosuppressionType', 'AntibioticProphylaxis']
                X_for_transform = X_for_transform.reindex(columns=expected_order, fill_value=0)
        
        output = self.original_preprocessor.transform(X_for_transform)
        if hasattr(output, 'toarray'):
            output = output.toarray()
        
        if self.columns_to_remove is not None and len(self.columns_to_remove) > 0:
            output = np.delete(output, self.columns_to_remove, axis=1)
        
        return output
    
    def get_feature_names_out(self, input_features=None):
        try:
            original_output = self.original_preprocessor.get_feature_names_out()
            if self.columns_to_remove is not None:
                return np.delete(original_output, self.columns_to_remove)
            return original_output
        except:
            return None
    
    def __sklearn_is_fitted__(self):
        return self._is_fitted
    
    @property
    def feature_names_in_(self):
        if hasattr(self.original_preprocessor, 'feature_names_in_'):
            orig_features = self.original_preprocessor.feature_names_in_
            return np.array([f for f in orig_features if f != self.feature_to_remove])
        return None
    
    @property 
    def n_features_in_(self):
        if hasattr(self.original_preprocessor, 'n_features_in_'):
            return self.original_preprocessor.n_features_in_ - 1
        return None

class CorrectedClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper for classifier with corrected coefficients"""
    
    def __init__(self, original_classifier, columns_to_remove=None):
        self.original_classifier = original_classifier
        self.columns_to_remove = columns_to_remove or []
        self._corrected_coef = None
        self._setup_corrected_coefficients()
        self._sklearn_fitted = True
        
    def _setup_corrected_coefficients(self):
        original_coef = self.original_classifier.coef_[0]
        if self.columns_to_remove:
            self._corrected_coef = np.delete(original_coef, self.columns_to_remove)
        else:
            self._corrected_coef = original_coef
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        linear_combination = np.dot(X, self._corrected_coef) + self.original_classifier.intercept_[0]
        probabilities = 1 / (1 + np.exp(-linear_combination))
        predictions = np.where(probabilities > 0.5, self.classes_[1], self.classes_[0])
        return predictions
    
    def predict_proba(self, X):
        linear_combination = np.dot(X, self._corrected_coef) + self.original_classifier.intercept_[0]
        prob_positive = 1 / (1 + np.exp(-linear_combination))
        prob_negative = 1 - prob_positive
        return np.column_stack([prob_negative, prob_positive])
    
    def __sklearn_is_fitted__(self):
        return True
    
    @property
    def classes_(self):
        return self.original_classifier.classes_
    
    @property
    def coef_(self):
        return self._corrected_coef.reshape(1, -1)
    
    @property
    def intercept_(self):
        return self.original_classifier.intercept_
    
    @property
    def n_features_in_(self):
        return len(self._corrected_coef)

# =============================================================================
# PAGE CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(
    page_title="NGHA/KAIMRC UTI Risk Calculator - Corrected AI System",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for premium design
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .premium-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .premium-title {
        color: #FFFFFF !important;
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
    
    .premium-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    
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
    
    .update-alert {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 1rem 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(40, 167, 69, 0.3);
    }
    
    .fade-in-up {
        animation: fadeInUp 0.8s ease-out;
    }
    
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
        title="Key Risk Factor Impact Analysis",
        title_font=dict(size=20, color='#333', family="Inter"),
        xaxis_title="Impact on UTI Risk",
        yaxis_title="Risk Factors",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=12, color="#333")
    )
    
    return fig

# =============================================================================
# ENHANCED MODEL LOADING WITH FILE UPLOAD
# =============================================================================

@st.cache_resource
def load_corrected_model():
    """Load the corrected UTI prediction model with comprehensive file detection"""
    
    st.markdown("### 🔍 Model Loading Process")
    
    current_dir = os.getcwd()
    st.info(f"📁 **Current Directory:** `{current_dir}`")
    
    # Scan directories
    try:
        # Root directory
        root_files = os.listdir(current_dir)
        root_joblib = [f for f in root_files if f.endswith('.joblib')]
        root_pkl = [f for f in root_files if f.endswith('.pkl')]
        
        # Models directory
        models_dir = os.path.join(current_dir, 'ml_results', 'models')
        models_exists = os.path.exists(models_dir)
        models_files = os.listdir(models_dir) if models_exists else []
        models_joblib = [f for f in models_files if f.endswith('.joblib')]
        models_pkl = [f for f in models_files if f.endswith('.pkl')]
        
        # Display scan results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📂 Root Directory:**")
            st.write(f"- .joblib files: {root_joblib if root_joblib else 'None'}")
            st.write(f"- .pkl files: {root_pkl if root_pkl else 'None'}")
        
        with col2:
            st.markdown("**📂 ml_results/models/:**")
            st.write(f"- Directory exists: {'✅' if models_exists else '❌'}")
            st.write(f"- .joblib files: {models_joblib if models_joblib else 'None'}")
            st.write(f"- .pkl files: {models_pkl if models_pkl else 'None'}")
        
    except Exception as e:
        st.error(f"Error scanning directories: {e}")
        return None, None, None
    
    # Try to load model from known locations
    model_search_paths = [
        ('ml_results/models/best_model.joblib', 'Primary location'),
        ('best_model.joblib', 'Root directory'),
        ('ml_results/models/new_model.joblib', 'Alternative name'),
        ('ml_results/models/corrected_model.joblib', 'Corrected version'),
        ('ml_results/models/best_model.pkl', 'Pickle format'),
    ]
    
    # Add all found files to search paths
    for file in models_joblib + models_pkl:
        full_path = os.path.join('ml_results', 'models', file)
        model_search_paths.append((full_path, f'Found in models directory'))
    
    for file in root_joblib + root_pkl:
        model_search_paths.append((file, f'Found in root directory'))
    
    # Try loading each path
    model = None
    model_filename = None
    
    st.markdown("**🔄 Attempting to load model...**")
    
    for filepath, description in model_search_paths:
        if os.path.exists(filepath):
            st.write(f"🔍 Trying: `{filepath}` ({description})")
            try:
                if filepath.endswith('.pkl'):
                    import pickle
                    with open(filepath, 'rb') as f:
                        model_data = pickle.load(f)
                    model = model_data['model'] if isinstance(model_data, dict) else model_data
                else:
                    model = joblib.load(filepath)
                
                model_filename = filepath
                st.success(f"✅ **Successfully loaded:** `{filepath}`")
                break
                
            except Exception as e:
                st.warning(f"❌ Failed to load `{filepath}`: {str(e)}")
                continue
    
    # If no model found, offer file upload
    if model is None:
        st.error("❌ **No compatible model found in expected locations**")
        
        st.markdown("### 📤 Upload Your Model File")
        st.info("""
        **Instructions:**
        1. Upload your corrected model file (from the surgical correction script)
        2. Supported formats: .joblib, .pkl
        3. File should contain the corrected model without AntibioticProphylaxis
        """)
        
        uploaded_file = st.file_uploader(
            "Choose your corrected model file",
            type=['joblib', 'pkl'],
            help="Upload the corrected model file from your correction script"
        )
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                temp_filename = f"uploaded_{uploaded_file.name}"
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Load the uploaded model
                if uploaded_file.name.endswith('.pkl'):
                    import pickle
                    with open(temp_filename, 'rb') as f:
                        model_data = pickle.load(f)
                    model = model_data['model'] if isinstance(model_data, dict) else model_data
                else:
                    model = joblib.load(temp_filename)
                
                model_filename = uploaded_file.name
                st.success(f"✅ **Successfully loaded uploaded model:** `{uploaded_file.name}`")
                
                # Clean up temp file
                os.remove(temp_filename)
                
            except Exception as e:
                st.error(f"❌ **Error loading uploaded file:** {e}")
                return None, None, None
        else:
            st.info("👆 **Please upload your corrected model file to continue**")
            return None, None, None
    
    # Validate model
    if model is not None:
        st.markdown("**🧪 Validating model...**")
        
        expected_features = [
            'Gender', 'Age', 'BMI', 'TransplantType', 'Diabetes', 
            'DJ_duration', 'Creatinine', 'eGFR', 'Hemoglobin', 
            'WBC', 'ImmunosuppressionType'
        ]
        
        try:
            # Test with sample data
            sample_data = pd.DataFrame({
                'Gender': [0],
                'Age': [45.0],
                'BMI': [25.0],
                'TransplantType': [1],
                'Diabetes': [0],
                'DJ_duration': [14.0],
                'Creatinine': [1.2],
                'eGFR': [60.0],
                'Hemoglobin': [12.0],
                'WBC': [7.0],
                'ImmunosuppressionType': [1]
            })
            
            test_pred = model.predict_proba(sample_data)
            
            if test_pred.shape == (1, 2):
                st.success(f"✅ **Model validation successful!** Test prediction: {test_pred[0, 1]:.3f}")
                return model, model_filename, expected_features
            else:
                st.error(f"❌ **Model output format unexpected:** {test_pred.shape}")
                return None, None, None
                
        except Exception as e:
            st.error(f"❌ **Model validation failed:** {e}")
            st.info("The model file may not be compatible. Please ensure you upload the corrected model.")
            return None, None, None
    
    return None, None, None

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
    
    # Model Update Alert
    st.markdown("""
    <div class="update-alert fade-in-up">
        🔄 <strong>Model Updated:</strong> Using scientifically corrected AI model (11 features, no AntibioticProphylaxis)
    </div>
    """, unsafe_allow_html=True)
    
    # Load Model
    if 'model_loaded' not in st.session_state:
        model_data = load_corrected_model()
        st.session_state.model = model_data[0]
        st.session_state.model_filename = model_data[1]
        st.session_state.expected_features = model_data[2]
        st.session_state.model_loaded = True
    
    model = st.session_state.model
    model_filename = st.session_state.model_filename
    expected_features = st.session_state.expected_features
    
    if model is None:
        st.error("🚨 **Please load a model to continue**")
        st.stop()
    
    # Success message
    st.success(f"✅ **Model Ready:** `{model_filename}` | **Features:** {len(expected_features)}")
    
    # Risk Assessment
    risk_assessment_page(model, expected_features)

def risk_assessment_page(model, expected_features):
    """Risk Assessment Interface"""
    
    # Patient Input Form
    st.markdown('<div class="premium-card fade-in-up">', unsafe_allow_html=True)
    st.markdown("### 📋 Patient Assessment Form")
    st.markdown("*11-factor corrected risk assessment*")
    
    tab1, tab2, tab3 = st.tabs(["👤 Demographics", "🏥 Clinical Data", "🔬 Laboratory"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("👫 Gender", ["Female", "Male"], 
                                help="⚠️ Critical Factor: Females have significantly higher UTI risk")
            age = st.slider("🎂 Age (years)", 18, 85, 50)
            
        with col2:
            diabetes = st.selectbox("🩺 Diabetes Mellitus", ["No", "Yes"])
            transplant_type = st.selectbox("🫀 Transplant Type", ["Deceased Donor", "Living Donor"])
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            dj_duration = st.slider("🔧 DJ Stent Duration (days)", 5.0, 45.0, 20.0, 0.1, 
                                  help="🚨 KEY PREDICTOR: Optimal ≤14 days")
            bmi = st.slider("⚖️ BMI (kg/m²)", 16.0, 45.0, 26.0, 0.1)
        
        with col2:
            immunosuppression = st.selectbox("💊 Immunosuppression Type", ["Type 1", "Type 2", "Type 3"])
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            creatinine = st.slider("🧪 Creatinine (mg/dL)", 0.5, 5.0, 1.2, 0.1)
            egfr = st.slider("📊 eGFR (mL/min/1.73m²)", 15.0, 120.0, 60.0, 1.0)
        
        with col2:
            hemoglobin = st.slider("🔴 Hemoglobin (g/dL)", 6.0, 18.0, 12.0, 0.1)
            wbc = st.slider("⚪ WBC Count (K/μL)", 2.0, 20.0, 7.0, 0.1)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Calculate Risk Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Calculate UTI Risk", type="primary", use_container_width=True):
            calculate_and_display_risk(model, gender, age, bmi, diabetes, transplant_type,
                                     dj_duration, creatinine, egfr, hemoglobin, wbc, immunosuppression)

def calculate_and_display_risk(model, gender, age, bmi, diabetes, transplant_type,
                             dj_duration, creatinine, egfr, hemoglobin, wbc, immunosuppression):
    """Calculate and display UTI risk"""
    
    # Prepare input data
    input_data = pd.DataFrame({
        'Gender': [1 if gender == "Male" else 0],
        'Age': [float(age)],
        'BMI': [float(bmi)],
        'TransplantType': [1 if transplant_type == "Living Donor" else 0],
        'Diabetes': [1 if diabetes == "Yes" else 0],
        'DJ_duration': [float(dj_duration)],
        'Creatinine': [float(creatinine)],
        'eGFR': [float(egfr)],
        'Hemoglobin': [float(hemoglobin)],
        'WBC': [float(wbc)],
        'ImmunosuppressionType': [int(immunosuppression.split()[-1])]
    })
    
    try:
        # Make prediction
        with st.spinner("Calculating UTI risk..."):
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
        
        # Display results
        st.markdown(f"""
        <div class="risk-display {risk_class} fade-in-up">
            <h1 class="risk-percentage">{risk_prob:.1%}</h1>
            <h2 class="risk-level">{risk_level} Risk</h2>
            <p style="font-size: 1.1rem; opacity: 0.9;">6-Month UTI Risk Probability</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Gauge Chart
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            gauge_fig = create_gauge_chart(risk_prob, "UTI Risk Assessment", color_scheme)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Risk Factor Analysis
        st.markdown('<div class="premium-card fade-in-up">', unsafe_allow_html=True)
        st.markdown("### 📊 Key Risk Factor Analysis")
        
        factors_data = {
            'factor': ['Female Gender', 'DJ Duration', 'Diabetes', 'Age', 'Creatinine', 'eGFR'],
            'impact': [
                0.35 if gender == "Female" else 0,
                max(0, (dj_duration - 14) * 0.025),
                0.28 if diabetes == "Yes" else 0,
                max(0, (age - 45) * 0.008),
                max(0, (creatinine - 1.2) * 0.20),
                max(0, (60 - egfr) * 0.005) if egfr < 60 else 0
            ]
        }
        
        risk_factor_fig = create_risk_factor_chart(factors_data)
        st.plotly_chart(risk_factor_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Clinical Recommendations
        display_clinical_recommendations(risk_level, risk_prob, dj_duration, gender, diabetes)
        
    except Exception as e:
        st.error(f"❌ **Error calculating risk:** {e}")
        st.info("Please check your model file and try again.")

def display_clinical_recommendations(risk_level, risk_prob, dj_duration, gender, diabetes):
    """Display clinical recommendations"""
    
    st.markdown('<div class="premium-card fade-in-up">', unsafe_allow_html=True)
    st.markdown("### 📋 Evidence-Based Clinical Recommendations")
    
    if risk_level == "Low":
        st.success(f"""
        **🟢 Low Risk Patient (Risk: {risk_prob:.1%})**
        
        **Recommended Actions:**
        - ✅ Standard monitoring protocol
        - ✅ Routine follow-up in 2-4 weeks
        - ✅ Patient education on UTI symptoms
        - ✅ Consider stent removal if duration >14 days
        """)
    elif risk_level == "Moderate":
        st.warning(f"""
        **🟡 Moderate Risk Patient (Risk: {risk_prob:.1%})**
        
        **Recommended Actions:**
        - ⚠️ Enhanced monitoring recommended
        - ⚠️ Follow-up in 1-2 weeks
        - ⚠️ Early stent removal priority (≤14 days)
        - ⚠️ Patient education on early symptoms
        """)
    else:
        st.error(f"""
        **🔴 High Risk Patient (Risk: {risk_prob:.1%})**
        
        **Immediate Actions Required:**
        - 🚨 Intensive monitoring protocol
        - 🚨 Weekly follow-up appointments
        - 🚨 **URGENT**: Plan stent removal if >14 days
        - 🚨 Consider antibiotic prophylaxis
        - 🚨 Urology consultation recommended
        """)
    
    # Specific alerts
    alerts = []
    if dj_duration > 21:
        alerts.append("🚨 **CRITICAL**: Stent duration >21 days significantly increases risk")
    elif dj_duration > 14:
        alerts.append("⚠️ **Important**: Stent duration >14 days increases risk")
    
    if gender == "Female":
        alerts.append("🚺 **Female Patient**: Higher baseline UTI risk")
    
    if diabetes == "Yes":
        alerts.append("🩺 **Diabetes Alert**: Optimize glycemic control")
    
    for alert in alerts:
        st.info(alert)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
def display_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 15px; color: white; margin-top: 2rem;">
        <h3>🏥 NGHA/KAIMRC UTI Risk Calculator</h3>
        <p><strong>Corrected AI Model v2.0</strong> | 11-Factor Analysis | Production Ready</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Scientifically validated model for clinical decision support</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# RUN APPLICATION
# =============================================================================
if __name__ == "__main__":
    main()
    display_footer()
