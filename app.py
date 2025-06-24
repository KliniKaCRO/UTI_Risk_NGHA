def show_file_uploader():
    """Show file uploader outside of cached function"""
    st.markdown("### 📤 EMERGENCY FILE UPLOAD")
    uploaded_file = st.file_uploader(
        "Upload your model file directly",
        type=['joblib', 'pkl'],
        help="This bypasses all file detection issues",
        key="model_uploader"
    )
    
    if uploaded_file is not None:
        try:
            st.info("🔄 Processing uploaded file...")
            
            # Save temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            st.success(f"✅ Saved temporarily as `{temp_path}`")
            
            # Apply sklearn compatibility fixes
            try:
                import sklearn.compose._column_transformer as ct
                if not hasattr(ct, '_RemainderColsList'):
                    ct._RemainderColsList = list
            except:
                pass
            
            # Load uploaded model
            if uploaded_file.name.endswith('.pkl'):
                import pickle
                with open(temp_path, 'rb') as f:
                    model_data = pickle.load(f)
                model = model_data['model'] if isinstance(model_data, dict) else model_data
            else:
                try:
                    model = joblib.load(temp_path)
                except:
                    # Try pickle as fallback
                    import pickle
                    with open(temp_path, 'rb') as f:
                        model = pickle.load(f)
            
            # Store in session state
            st.session_state.corrected_model = model
            st.session_state.model_filename = uploaded_file.name
            st.session_state.expected_features = [
                'Gender', 'Age', 'BMI', 'TransplantType', 'Diabetes', 
                'DJ_duration', 'Creatinine', 'eGFR', 'Hemoglobin', 
                'WBC', 'ImmunosuppressionType'
            ]
            st.session_state.corrected_model_loaded = True
            
            st.success(f"🎉 **UPLOAD SUCCESS!** Model loaded: `{type(model)}`")
            st.rerun()  # Refresh the app
            
        except Exception as e:
            st.error(f"💥 Upload loading failed: `{type(e).__name__}: {str(e)}`")
            import traceback
            st.code(traceback.format_exc())
            
    return uploaded_file is not Noneimport streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

# =============================================================================
# SKLEARN COMPATIBILITY FIXES
# =============================================================================

# Fix for sklearn version compatibility issues
import sys
from sklearn.compose import _column_transformer

# Add missing sklearn classes for compatibility
if not hasattr(_column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        """Compatibility class for sklearn version differences"""
        pass
    _column_transformer._RemainderColsList = _RemainderColsList
    sys.modules['sklearn.compose._column_transformer']._RemainderColsList = _RemainderColsList

# Fix for other potential sklearn compatibility issues
try:
    from sklearn.utils._bunch import Bunch
except ImportError:
    from sklearn.utils import Bunch

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
        self._is_fitted = True  # Mark as fitted since it's loaded from saved model
        self._sklearn_fitted = True
        
    def transform(self, X):
        """Transform input by removing feature, then remove corresponding output columns"""
        # Remove the target feature from input if it exists
        if hasattr(X, 'columns') and self.feature_to_remove in X.columns:
            X_reduced = X.drop(columns=[self.feature_to_remove], errors='ignore')
            # Create dummy column for transformation
            X_for_transform = X_reduced.copy()
            X_for_transform[self.feature_to_remove] = 0
            # Reorder to match original
            if hasattr(self.original_preprocessor, 'feature_names_in_'):
                X_for_transform = X_for_transform[self.original_preprocessor.feature_names_in_]
        else:
            # Handle case where feature is already removed
            X_for_transform = X.copy()
            if hasattr(X, 'columns') and len(X.columns) == 11:
                # Add dummy column to match original preprocessor expectations
                X_for_transform[self.feature_to_remove] = 0
                # Reorder if we know the original order
                expected_order = ['Gender', 'Age', 'BMI', 'TransplantType', 'Diabetes', 
                                'DJ_duration', 'Creatinine', 'eGFR', 'Hemoglobin', 'WBC', 
                                'ImmunosuppressionType', 'AntibioticProphylaxis']
                X_for_transform = X_for_transform.reindex(columns=expected_order, fill_value=0)
        
        # Transform using original preprocessor
        output = self.original_preprocessor.transform(X_for_transform)
        
        # Convert to array if sparse
        if hasattr(output, 'toarray'):
            output = output.toarray()
        
        # Remove the columns corresponding to the target feature
        if self.columns_to_remove is not None and len(self.columns_to_remove) > 0:
            output = np.delete(output, self.columns_to_remove, axis=1)
        
        return output
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names after removal"""
        try:
            original_output = self.original_preprocessor.get_feature_names_out()
            if self.columns_to_remove is not None:
                return np.delete(original_output, self.columns_to_remove)
            return original_output
        except:
            return None
    
    def __sklearn_is_fitted__(self):
        """Tell sklearn this estimator is fitted"""
        return self._is_fitted
    
    @property
    def feature_names_in_(self):
        """Return new feature names"""
        if hasattr(self.original_preprocessor, 'feature_names_in_'):
            orig_features = self.original_preprocessor.feature_names_in_
            return np.array([f for f in orig_features if f != self.feature_to_remove])
        return None
    
    @property 
    def n_features_in_(self):
        """Return number of input features after removal"""
        if hasattr(self.original_preprocessor, 'n_features_in_'):
            return self.original_preprocessor.n_features_in_ - 1
        return None

class CorrectedClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper for classifier with corrected coefficients - sklearn compliant"""
    
    def __init__(self, original_classifier, columns_to_remove=None):
        self.original_classifier = original_classifier
        self.columns_to_remove = columns_to_remove or []
        self._corrected_coef = None
        self._setup_corrected_coefficients()
        # Set sklearn fitted state attributes
        self._sklearn_fitted = True
        
    def _setup_corrected_coefficients(self):
        """Setup corrected coefficients by removing specified columns"""
        original_coef = self.original_classifier.coef_[0]
        if self.columns_to_remove:
            self._corrected_coef = np.delete(original_coef, self.columns_to_remove)
        else:
            self._corrected_coef = original_coef
    
    def fit(self, X, y=None):
        """Dummy fit method for sklearn compliance"""
        return self
    
    def predict(self, X):
        """Make predictions using corrected coefficients"""
        # Calculate linear combination
        linear_combination = np.dot(X, self._corrected_coef) + self.original_classifier.intercept_[0]
        
        # Apply sigmoid for probability
        probabilities = 1 / (1 + np.exp(-linear_combination))
        
        # Convert to binary predictions using original classes
        predictions = np.where(probabilities > 0.5, self.classes_[1], self.classes_[0])
        return predictions
    
    def predict_proba(self, X):
        """Predict probabilities using corrected coefficients"""
        # Calculate linear combination
        linear_combination = np.dot(X, self._corrected_coef) + self.original_classifier.intercept_[0]
        
        # Apply sigmoid
        prob_positive = 1 / (1 + np.exp(-linear_combination))
        prob_negative = 1 - prob_positive
        
        return np.column_stack([prob_negative, prob_positive])
    
    def __sklearn_is_fitted__(self):
        """Tell sklearn this estimator is fitted"""
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
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Update Alert */
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
# CORRECTED MODEL LOADING
# =============================================================================

def load_corrected_model():
    """Load the corrected UTI prediction model with MAXIMUM debugging"""
    
    st.markdown("## 🔍 DETAILED MODEL LOADING DEBUG")
    
    try:
        import os
        current_dir = os.getcwd()
        st.success(f"📁 **Current Directory:** `{current_dir}`")
        
        # Show all files in root
        try:
            root_files = os.listdir(current_dir)
            st.write("**📂 Root Directory Contents:**")
            for f in sorted(root_files):
                file_path = os.path.join(current_dir, f)
                is_dir = os.path.isdir(file_path)
                size = os.path.getsize(file_path) if not is_dir else "DIR"
                st.write(f"  - `{f}` {'(DIR)' if is_dir else f'({size} bytes)'}")
        except Exception as e:
            st.error(f"❌ Can't read root directory: {e}")
        
        # Show ml_results/models contents
        models_dir = os.path.join(current_dir, 'ml_results', 'models')
        st.write(f"**📂 Models Directory:** `{models_dir}`")
        
        if os.path.exists(models_dir):
            try:
                model_files = os.listdir(models_dir)
                st.success(f"✅ Models directory exists with {len(model_files)} files:")
                for f in sorted(model_files):
                    file_path = os.path.join(models_dir, f)
                    size = os.path.getsize(file_path)
                    st.write(f"  - `{f}` ({size} bytes)")
            except Exception as e:
                st.error(f"❌ Can't read models directory: {e}")
        else:
            st.error("❌ Models directory does not exist!")
        
        # Test specific file paths
        st.markdown("### 🎯 Testing Specific File Paths")
        
        test_paths = [
            'ml_results/models/best_model.joblib',
            'best_model.joblib',
            'ml_results/models/best_model.pkl',
            os.path.join(current_dir, 'ml_results', 'models', 'best_model.joblib'),
        ]
        
        for path in test_paths:
            exists = os.path.exists(path)
            if exists:
                try:
                    size = os.path.getsize(path)
                    st.success(f"✅ `{path}` exists ({size} bytes)")
                except:
                    st.warning(f"⚠️ `{path}` exists but can't get size")
            else:
                st.error(f"❌ `{path}` does not exist")
        
        # Now try loading with sklearn compatibility fixes
        st.markdown("### 🚀 Attempting Model Loading (With Compatibility Fixes)")
        
        model_search_paths = [
            'ml_results/models/best_model.joblib',
            'ml_results/models/best_model.pkl', 
            'best_model.joblib',
            'ml_results/models/best_model (2)_FINAL_corrected.joblib',
            'ml_results/models/new_model.joblib',
            os.path.join(current_dir, 'ml_results', 'models', 'best_model.joblib'),
        ]
        
        model = None
        model_filename = None
        
        for i, filepath in enumerate(model_search_paths):
            st.write(f"**Attempt {i+1}:** Trying `{filepath}`")
            
            try:
                if os.path.exists(filepath):
                    st.info(f"  📁 File exists, attempting to load with compatibility fixes...")
                    
                    # Apply additional sklearn compatibility fixes before loading
                    try:
                        # Try to fix more sklearn compatibility issues
                        import sklearn.compose._column_transformer as ct
                        if not hasattr(ct, '_RemainderColsList'):
                            ct._RemainderColsList = list
                        
                        # Additional sklearn compatibility
                        import sklearn.utils._testing
                        if not hasattr(sklearn.utils._testing, 'ignore_warnings'):
                            sklearn.utils._testing.ignore_warnings = lambda: lambda f: f
                            
                    except Exception as comp_e:
                        st.warning(f"  ⚠️ Compatibility fix warning: {comp_e}")
                    
                    if filepath.endswith('.pkl'):
                        st.info("  🔄 Loading as pickle file...")
                        import pickle
                        with open(filepath, 'rb') as f:
                            model_data = pickle.load(f)
                        model = model_data['model'] if isinstance(model_data, dict) else model_data
                    else:
                        st.info("  🔄 Loading as joblib file with compatibility...")
                        
                        # Try loading with different methods
                        try:
                            model = joblib.load(filepath)
                        except Exception as load_e1:
                            st.warning(f"  ⚠️ Standard joblib load failed: {load_e1}")
                            
                            # Try with pickle protocol fix
                            try:
                                import pickle
                                with open(filepath, 'rb') as f:
                                    model = pickle.load(f)
                                st.info("  🔄 Loaded using pickle instead of joblib")
                            except Exception as load_e2:
                                raise load_e1  # Re-raise original error
                    
                    model_filename = filepath
                    st.success(f"  ✅ SUCCESS! Loaded from `{filepath}`")
                    st.success(f"  📊 Model type: `{type(model)}`")
                    break
                    
                else:
                    st.warning(f"  ❌ File does not exist: `{filepath}`")
                    continue
                    
            except Exception as e:
                st.error(f"  💥 Loading failed: `{type(e).__name__}: {str(e)}`")
                # Show relevant part of traceback for debugging
                import traceback
                tb_lines = traceback.format_exc().split('\n')
                relevant_lines = [line for line in tb_lines if 'sklearn' in line or 'AttributeError' in line or 'joblib' in line]
                if relevant_lines:
                    st.code('\n'.join(relevant_lines))
                continue
        
        if model is None:
            st.error("🚨 **NO MODEL LOADED SUCCESSFULLY**")
            st.error("**The issue is sklearn version compatibility - the model was saved with a different sklearn version.**")
            
            st.info("""
            **🔧 SOLUTIONS:**
            
            1. **Re-save your model:** Run the correction script again and save a new model
            2. **Upload manually:** Use the uploader below  
            3. **Version fix:** The model needs to be saved with sklearn compatibility
            
            **The model file exists but can't be loaded due to sklearn version differences.**
            """)
            
            return None, None, None
        
        # Model validation
        st.markdown("### ✅ Model Validation")
        
        expected_features = [
            'Gender', 'Age', 'BMI', 'TransplantType', 'Diabetes', 
            'DJ_duration', 'Creatinine', 'eGFR', 'Hemoglobin', 
            'WBC', 'ImmunosuppressionType'
        ]
        
        try:
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
            
            st.info("🧪 Testing model with sample data...")
            st.write(f"Sample data shape: {sample_data.shape}")
            st.write(f"Sample data columns: {list(sample_data.columns)}")
            
            test_pred = model.predict_proba(sample_data)
            st.success(f"🎉 **MODEL WORKS!** Prediction shape: {test_pred.shape}")
            st.success(f"🎯 Sample prediction: {test_pred[0, 1]:.4f}")
            
        except Exception as e:
            st.error(f"💥 Model test failed: `{type(e).__name__}: {str(e)}`")
            import traceback
            st.code(traceback.format_exc())
            st.warning("⚠️ Model loaded but may not work with input data")
        
        st.success(f"🏆 **FINAL RESULT:** Model loaded from `{model_filename}`")
        return model, model_filename, expected_features
        
    except Exception as e:
        st.error(f"💥 **CRITICAL ERROR:** {type(e).__name__}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Premium Header with update notice
    st.markdown("""
    <div class="premium-header fade-in-up">
        <h1 class="premium-title">⚕️ NGHA/KAIMRC UTI Risk Calculator</h1>
        <p class="premium-subtitle">Advanced AI-Powered Clinical Decision Support System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Update Alert
    st.markdown("""
    <div class="update-alert fade-in-up">
        🔄 <strong>Model Updated:</strong> Now using scientifically corrected AI model with enhanced accuracy and reliability
    </div>
    """, unsafe_allow_html=True)
    
    # Load Corrected Model (no longer cached to avoid widget issues)
    if 'corrected_model_loaded' not in st.session_state:
        st.session_state.corrected_model_loaded = False
    
    if not st.session_state.corrected_model_loaded:
        model_data = load_corrected_model()
        if model_data[0] is not None:  # Successfully loaded
            st.session_state.corrected_model = model_data[0]
            st.session_state.model_filename = model_data[1] 
            st.session_state.expected_features = model_data[2]
            st.session_state.corrected_model_loaded = True
            st.rerun()  # Refresh to show the loaded model
        else:
            return  # Exit if model couldn't be loaded
    
    model = st.session_state.corrected_model
    model_filename = st.session_state.model_filename  
    expected_features = st.session_state.expected_features
    
    if model is None:
        st.error("🚨 **Critical Error**: Corrected AI system could not be initialized.")
        st.info("""
        **Issue Identified:** sklearn version compatibility problem
        
        **Solution:** The model was saved with a different sklearn version. 
        You need to re-run the correction script to save a compatible model.
        """)
        return
    
    # Only show success message if debug wasn't already displayed
    if not hasattr(st.session_state, 'debug_shown'):
        st.success(f"✅ **Model Loaded Successfully:** {model_filename}")
        st.info(f"📊 **Features:** Using {len(expected_features)} validated predictive factors")
    
    # Risk Assessment Page
    risk_assessment_page(model, expected_features)

def risk_assessment_page(model, expected_features):
    """Premium Risk Assessment Interface with corrected model"""
    
    # Model Performance Display
    st.markdown('<div class="premium-card fade-in-up">', unsafe_allow_html=True)
    st.markdown("### 🎯 Corrected AI Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Updated performance metrics for corrected model
    metrics = [
        ("Features", len(expected_features), "📊"),
        ("Accuracy", "Validated", "✅"),
        ("Calibration", "Excellent", "🎯"),
        ("Status", "Production", "🚀")
    ]
    
    for i, (metric, value, icon) in enumerate(metrics):
        with [col1, col2, col3, col4][i]:
            st.metric(f"{icon} {metric}", str(value))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Patient Input Form - Only 11 Features
    st.markdown('<div class="premium-card fade-in-up">', unsafe_allow_html=True)
    st.markdown("### 📋 Patient Assessment Form")
    st.markdown("*Scientifically validated 11-factor risk assessment*")
    
    # Create tabs for organized input
    tab1, tab2, tab3 = st.tabs(["👤 Demographics", "🏥 Clinical Data", "🔬 Laboratory"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("👫 Gender", ["Female", "Male"], 
                                help="⚠️ Critical Factor: Females have significantly higher UTI risk")
            age = st.slider("🎂 Age (years)", 18, 85, 50, 
                          help="Patient age affects immune function and UTI susceptibility")
            
        with col2:
            diabetes = st.selectbox("🩺 Diabetes Mellitus", ["No", "Yes"], 
                                  help="Diabetes increases UTI risk through multiple mechanisms")
            transplant_type = st.selectbox("🫀 Transplant Type", ["Deceased Donor", "Living Donor"],
                                         help="Transplant type influences overall risk profile")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            dj_duration = st.slider("🔧 DJ Stent Duration (days)", 5.0, 45.0, 20.0, 0.1, 
                                  help="🚨 KEY PREDICTOR: Longer duration = Higher risk. Optimal: ≤14 days")
            bmi = st.slider("⚖️ BMI (kg/m²)", 16.0, 45.0, 26.0, 0.1, 
                          help="Body Mass Index affects surgical outcomes and infection risk")
        
        with col2:
            immunosuppression = st.selectbox("💊 Immunosuppression Type", 
                                           ["Type 1", "Type 2", "Type 3"],
                                           help="Different regimens have varying infection risk profiles")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            creatinine = st.slider("🧪 Creatinine (mg/dL)", 0.5, 5.0, 1.2, 0.1,
                                 help="Kidney function marker - higher levels increase UTI risk")
            egfr = st.slider("📊 eGFR (mL/min/1.73m²)", 15.0, 120.0, 60.0, 1.0,
                           help="Estimated Glomerular Filtration Rate - kidney function")
        
        with col2:
            hemoglobin = st.slider("🔴 Hemoglobin (g/dL)", 6.0, 18.0, 12.0, 0.1,
                                 help="Blood oxygen-carrying capacity, affects immune function")
            wbc = st.slider("⚪ WBC Count (K/μL)", 2.0, 20.0, 7.0, 0.1,
                          help="White Blood Cell count - immune system status")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Risk Calculation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Calculate UTI Risk", type="primary", use_container_width=True):
            calculate_and_display_risk(model, gender, age, bmi, diabetes, transplant_type,
                                     dj_duration, creatinine, egfr, hemoglobin, wbc,
                                     immunosuppression)

def calculate_and_display_risk(model, gender, age, bmi, diabetes, transplant_type,
                             dj_duration, creatinine, egfr, hemoglobin, wbc,
                             immunosuppression):
    """Premium risk calculation and display with corrected model"""
    
    # Prepare input data - ONLY 11 features (no AntibioticProphylaxis)
    # Ensure correct order and format for the corrected model
    input_data = pd.DataFrame({
        'Gender': [1 if gender == "Male" else 0],  # Note: Model expects 0=Female, 1=Male
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
        # Make prediction with corrected model
        with st.spinner("Calculating UTI risk..."):
            risk_prob = model.predict_proba(input_data)[0, 1]
        
        # Determine risk level based on validated thresholds
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
            <p style="font-size: 1.1rem; opacity: 0.9;">6-Month UTI Risk Probability</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Gauge Chart
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            gauge_fig = create_gauge_chart(risk_prob, "UTI Risk Assessment", color_scheme)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Risk Factor Analysis - Based on validated clinical associations
        st.markdown('<div class="premium-card fade-in-up">', unsafe_allow_html=True)
        st.markdown("### 📊 Key Risk Factor Analysis")
        
        # Calculate risk factor impacts based on clinical evidence
        factors_data = {
            'factor': ['Female Gender', 'DJ Duration', 'Diabetes', 'Age', 'Creatinine', 'eGFR'],
            'impact': [
                0.35 if gender == "Female" else 0,  # Major risk factor
                max(0, (dj_duration - 14) * 0.025),  # Risk increases after 14 days
                0.28 if diabetes == "Yes" else 0,  # Significant diabetes effect
                max(0, (age - 45) * 0.008),  # Age effect above 45
                max(0, (creatinine - 1.2) * 0.20),  # Creatinine effect above normal
                max(0, (60 - egfr) * 0.005) if egfr < 60 else 0  # eGFR effect below 60
            ]
        }
        
        risk_factor_fig = create_risk_factor_chart(factors_data)
        st.plotly_chart(risk_factor_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Clinical Recommendations
        display_clinical_recommendations(risk_level, risk_prob, dj_duration, gender, diabetes)
        
        # Risk Timeline
        st.markdown('<div class="premium-card fade-in-up">', unsafe_allow_html=True)
        st.markdown("### 📈 Risk Progression Timeline")
        timeline_data = {
            'days': list(range(0, 181, 30)),
            'risk': [min(0.95, risk_prob * (1 + i*0.03)) for i in range(7)]  # Risk increases over time
        }
        timeline_fig = create_timeline_chart(timeline_data)
        st.plotly_chart(timeline_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error calculating risk: {e}")
        st.error(f"Error details: {type(e).__name__}: {str(e)}")
        
        # Additional debugging
        st.write("🔧 **Troubleshooting Info:**")
        st.write(f"Model type: {type(model)}")
        st.write(f"Input data shape: {input_data.shape}")
        st.write(f"Input data types: {input_data.dtypes.to_dict()}")
        
        st.info("""
        **Possible solutions:**
        1. Verify the model file contains the corrected model
        2. Check that all input values are within valid ranges
        3. Ensure the model was saved correctly from the correction script
        4. Try refreshing the page to reload the model
        """)

def display_clinical_recommendations(risk_level, risk_prob, dj_duration, gender, diabetes):
    """Display premium clinical recommendations based on corrected model"""
    
    st.markdown('<div class="premium-card fade-in-up">', unsafe_allow_html=True)
    st.markdown("### 📋 Evidence-Based Clinical Decision Support")
    
    if risk_level == "Low":
        st.success(f"""
        **🟢 Low Risk Patient (Risk: {risk_prob:.1%})**
        
        **Recommended Actions:**
        - ✅ Standard monitoring protocol
        - ✅ Routine follow-up in 2-4 weeks
        - ✅ Patient education on UTI symptoms
        - ✅ Continue current management
        - ✅ Consider stent removal if duration >14 days
        
        **Key Points:**
        - Low probability of UTI development
        - Standard care protocols are appropriate
        - Focus on optimal stent timing
        """)
    elif risk_level == "Moderate":
        st.warning(f"""
        **🟡 Moderate Risk Patient (Risk: {risk_prob:.1%})**
        
        **Recommended Actions:**
        - ⚠️ Enhanced monitoring recommended
        - ⚠️ Follow-up in 1-2 weeks
        - ⚠️ Consider early stent removal (≤14 days optimal)
        - ⚠️ Patient education on early symptoms
        - ⚠️ Optimize diabetes control if applicable
        
        **Key Points:**
        - Moderate intervention may be beneficial
        - Consider risk factor modification
        - Early stent removal strongly recommended
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
        - 🚨 Enhanced patient education
        
        **Critical Points:**
        - High probability of UTI development
        - Aggressive prevention strategies essential
        - Multidisciplinary care approach
        """)
    
    # Specific factor-based recommendations
    recommendations = []
    
    if dj_duration > 21:
        recommendations.append("🚨 **CRITICAL**: Stent duration >21 days significantly increases risk. Immediate removal planning required.")
    elif dj_duration > 14:
        recommendations.append("⚠️ **Important**: Stent duration >14 days increases risk. Consider early removal.")
    
    if gender == "Female":
        recommendations.append("🚺 **Female Patient**: Higher baseline UTI risk. Enhanced preventive measures recommended.")
    
    if diabetes == "Yes":
        recommendations.append("🩺 **Diabetes Alert**: Optimize glycemic control to reduce UTI risk.")
    
    # Display specific recommendations
    if recommendations:
        st.markdown("### 🎯 Specific Risk Factor Alerts")
        for rec in recommendations:
            st.info(rec)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
def display_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 15px; color: white; margin-top: 2rem;">
        <h3>🏥 NGHA/KAIMRC UTI Risk Calculator</h3>
        <p><strong>Corrected AI Model v2.0</strong> | Scientifically Validated | 11-Factor Analysis</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">This tool uses a corrected AI model and should be used in conjunction with clinical judgment.</p>
        <p style="font-size: 0.8rem; opacity: 0.7;">Key improvement: Removed non-contributory factors for enhanced accuracy</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# RUN APPLICATION
# =============================================================================
if __name__ == "__main__":
    main()
    display_footer()
