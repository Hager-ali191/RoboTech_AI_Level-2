# main.py - Machine Learning Professional Dashboard with YOUR Class Integration
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
import warnings
import time
from datetime import datetime
from pathlib import Path
import sys
import os

# Add your class imports
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
from sklearn.feature_selection import f_classif, chi2, SelectKBest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    confusion_matrix, classification_report
)
warnings.filterwarnings('ignore')

# Import YOUR MachineLearning class
from credit_score_project import MachineLearning

# Page configuration
st.set_page_config(
    page_title="ML Professional Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add this at the beginning of your app.py, right after st.set_page_config
# Custom CSS with Ocean Dark Theme
# Replace your CSS with this - ONLY text is white, everything else is dark

st.markdown("""
<style>
    /* Main Ocean Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0a192f 0%, #0c1f3d 50%, #10264d 100%) !important;
    }
    
    /* ========== TEXT COLORS ========== */
    /* Ensure ALL text is white */
    h1, h2, h3, h4, h5, h6, p, span, div, label, li, a, strong, em {
        color: #ffffff !important;
    }
    
    /* ========== TOP NAVIGATION ========== */
    .top-nav {
        background: linear-gradient(90deg, #051322 0%, #0a1a2d 50%, #0c1f3d 100%) !important;
        padding: 1rem 2rem;
        position: sticky;
        top: 0;
        z-index: 1000;
        border-bottom: 3px solid #64ffda !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5) !important;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        margin-bottom: 2rem;
    }
    
    .nav-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
    }
    
    .project-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #64ffda, #52d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .nav-buttons {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        justify-content: center;
        width: 100%;
    }
    
    /* Navigation buttons - DARK BACKGROUND */
    .stButton > button {
        background: linear-gradient(135deg, #112240 0%, #1a365d 100%) !important;
        color: #ffffff !important;
        border: 2px solid #64ffda !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        min-width: 140px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1a365d 0%, #112240 100%) !important;
        border-color: #52d4ff !important;
    }
    
    /* ========== CARDS AND CONTAINERS ========== */
    /* All cards have DARK backgrounds */
    .content-card {
        background: linear-gradient(135deg, rgba(17, 34, 64, 0.95) 0%, rgba(26, 54, 93, 0.95) 100%) !important;
        border: 1px solid #2d4a80 !important;
        color: #ffffff !important;
    }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(42, 74, 128, 0.9) 0%, rgba(60, 100, 170, 0.9) 100%) !important;
        border: 2px solid #64ffda !important;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(12, 31, 61, 0.9) 0%, rgba(17, 34, 64, 0.9) 100%) !important;
        border: 2px solid #64ffda !important;
    }
    
    /* ========== DATAFRAMES AND TABLES ========== */
    /* Force ALL tables to have dark backgrounds */
    table, .stDataFrame, .stTable, .dataframe {
        background-color: rgba(12, 31, 61, 0.9) !important;
        border: 1px solid #2d4a80 !important;
    }
    
    th, td {
        background-color: rgba(12, 31, 61, 0.9) !important;
        color: #ffffff !important;
        border-color: #2d4a80 !important;
    }
    
    tr {
        background-color: rgba(12, 31, 61, 0.9) !important;
    }
    
    /* ========== INPUT FIELDS ========== */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div,
    .stTextArea > div > div > textarea {
        background-color: rgba(12, 31, 61, 0.9) !important;
        color: #ffffff !important;
        border: 1px solid #64ffda !important;
    }
    
    /* ========== SELECT BOXES ========== */
    /* Make dropdown menus dark */
    [data-baseweb="select"] > div,
    [data-baseweb="popover"] > div,
    [data-baseweb="menu"] {
        background-color: rgba(12, 31, 61, 0.95) !important;
        border: 1px solid #64ffda !important;
    }
    
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] div[role="option"] {
        background-color: rgba(12, 31, 61, 0.95) !important;
        color: #ffffff !important;
    }
    
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] div[role="option"]:hover {
        background-color: rgba(26, 54, 93, 0.95) !important;
    }
    
    /* ========== METRICS ========== */
    .stMetric {
        background-color: rgba(12, 31, 61, 0.7) !important;
        border: 1px solid #64ffda !important;
    }
    
    .stMetric label {
        color: #a8b2d1 !important;
    }
    
    .stMetric div {
        color: #64ffda !important;
    }
    
    /* ========== EXPANDERS ========== */
    .streamlit-expanderHeader {
        background-color: rgba(12, 31, 61, 0.9) !important;
        color: #64ffda !important;
        border: 1px solid #64ffda !important;
    }
    
    .streamlit-expanderContent {
        background-color: rgba(12, 31, 61, 0.8) !important;
    }
    
    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(12, 31, 61, 0.8) !important;
        border: 1px solid rgba(100, 255, 218, 0.3) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(17, 34, 64, 0.8) !important;
        color: #a8b2d1 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(100, 255, 218, 0.2) !important;
        color: #64ffda !important;
        border: 2px solid #64ffda !important;
    }
    
    /* ========== SLIDERS ========== */
    .stSlider > div > div > div {
        background-color: rgba(12, 31, 61, 0.9) !important;
    }
    
    /* ========== RADIO BUTTONS ========== */
    .stRadio > div {
        background-color: rgba(12, 31, 61, 0.9) !important;
        border: 1px solid #64ffda !important;
    }
    
    /* ========== CHECKBOXES ========== */
    .stCheckbox > div {
        background-color: rgba(12, 31, 61, 0.9) !important;
    }
    
    /* ========== PLOT CONTAINERS ========== */
    .plot-container {
        background: rgba(12, 31, 61, 0.8) !important;
        border: 1px solid rgba(100, 255, 218, 0.3) !important;
    }
    
    /* ========== SUCCESS/WARNING/INFO CARDS ========== */
    .success-card {
        background: linear-gradient(135deg, rgba(27, 94, 32, 0.9) 0%, rgba(46, 125, 50, 0.9) 100%) !important;
        border: 1px solid #4caf50 !important;
    }
    
    .warning-card {
        background: linear-gradient(135deg, rgba(230, 81, 0, 0.9) 0%, rgba(255, 152, 0, 0.9) 100%) !important;
        border: 1px solid #ff9800 !important;
    }
    
    .danger-card {
        background: linear-gradient(135deg, rgba(183, 28, 28, 0.9) 0%, rgba(211, 47, 47, 0.9) 100%) !important;
        border: 1px solid #f44336 !important;
    }
    
    .info-card {
        background: linear-gradient(135deg, rgba(1, 87, 155, 0.9) 0%, rgba(2, 119, 189, 0.9) 100%) !important;
        border: 1px solid #4fc3f7 !important;
    }
    
    /* ========== STATUS INDICATORS ========== */
    .progress-step {
        background: rgba(26, 54, 93, 0.6) !important;
    }
    
    /* ========== REMOVE ANY WHITE BACKGROUNDS ========== */
    /* This removes ALL white backgrounds from Streamlit components */
    div[data-testid="stVerticalBlock"] > div,
    div[data-testid="stHorizontalBlock"] > div,
    .element-container,
    .st-emotion-cache-1jicfl2,
    .st-emotion-cache-1v0mbdj {
        background-color: transparent !important;
    }
    
    /* Remove any remaining white backgrounds */
    * {
        --background-color: transparent !important;
    }
    
    /* ========== THEME TOGGLE ========== */
    .theme-toggle {
        background: linear-gradient(135deg, #112240 0%, #1a365d 100%) !important;
        border: 2px solid #64ffda !important;
        color: #ffffff !important;
    }
    
    /* ========== SCROLLBAR ========== */
    ::-webkit-scrollbar-track {
        background: #0c1f3d !important;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #64ffda, #52d4ff) !important;
    }
</style>
""", unsafe_allow_html=True)

class MLDashboard:
    def __init__(self):
        self.ml = MachineLearning()
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            'df': None,
            'df_processed': None,
            'model': None,
            'feature_names': None,
            'x_train': None,
            'x_test': None,
            'y_train': None,
            'y_test': None,
            'current_step': 1,
            'data_loaded': False,
            'analysis_complete': False,
            'preprocessing_complete': False,
            'model_trained': False,
            'predictions_made': False,
            'training_history': [],
            'selected_columns': [],
            'plot_history': [],
            'insights': []
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def save_plot(self, fig, plot_name):
        """Save plot to session state history"""
        plot_data = {
            'name': plot_name,
            'timestamp': datetime.now(),
            'type': type(fig).__name__
        }
        st.session_state.plot_history.append(plot_data)
        
    def add_insight(self, title, content):
        """Add insight to session state"""
        insight = {
            'title': title,
            'content': content,
            'timestamp': datetime.now(),
            'step': st.session_state.current_step
        }
        st.session_state.insights.append(insight)
    
    def create_top_navigation(self):
        """Create top navigation bar centered above content"""
        st.markdown("""
        <div class="top-nav">
            <div class="nav-container">
                <div class="project-title">🤖 ML Professional Dashboard</div>
                <div class="nav-buttons">
        """, unsafe_allow_html=True)
        
        # Create columns for the navigation buttons
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        pages = [
            ("🏠 Dashboard", self.render_dashboard),
            ("📁 Data Loader", self.render_data_loader),
            ("🔍 Data Analysis", self.render_data_analysis),
            ("⚙️ Preprocessing", self.render_preprocessing),
            ("🤖 Model Training", self.render_model_training),
            ("🎯 Predictions", self.render_predictions)
        ]
        
        buttons = []
        with col1:
            if st.button("🏠 Dashboard", key="nav_dashboard", use_container_width=True):
                st.session_state.current_page = "🏠 Dashboard"
                st.rerun()
        with col2:
            if st.button("📁 Data Loader", key="nav_loader", use_container_width=True):
                st.session_state.current_page = "📁 Data Loader"
                st.rerun()
        with col3:
            if st.button("🔍 Data Analysis", key="nav_analysis", use_container_width=True):
                st.session_state.current_page = "🔍 Data Analysis"
                st.rerun()
        with col4:
            if st.button("⚙️ Preprocessing", key="nav_preprocessing", use_container_width=True):
                st.session_state.current_page = "⚙️ Preprocessing"
                st.rerun()
        with col5:
            if st.button("🤖 Model Training", key="nav_training", use_container_width=True):
                st.session_state.current_page = "🤖 Model Training"
                st.rerun()
        with col6:
            if st.button("🎯 Predictions", key="nav_predictions", use_container_width=True):
                st.session_state.current_page = "🎯 Predictions"
                st.rerun()
        
        st.markdown("""
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_dashboard(self):
        """Render main dashboard"""
        st.markdown('<h1 class="main-header">🤖 Machine Learning Professional Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Welcome section
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-header">🚀 Welcome to ML Dashboard</h2>', unsafe_allow_html=True)
            st.markdown("""
            <div class="insight-content">
            <p>This professional dashboard provides a complete machine learning workflow using <strong>YOUR MachineLearning class</strong>.</p>
            <p><strong>Features:</strong></p>
            <ul>
                <li>✅ Complete data analysis with interactive visualizations</li>
                <li>✅ Comprehensive preprocessing pipeline</li>
                <li>✅ Model training with hyperparameter tuning</li>
                <li>✅ Real-time predictions</li>
                <li>✅ Detailed insights and explanations</li>
            </ul>
            <p><strong>Get Started:</strong> Load your dataset using the Data Loader page.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">⚡ Quick Actions</h3>', unsafe_allow_html=True)
            
            if st.button("📁 Load Sample Data", use_container_width=True):
                self.load_sample_data()
                st.success("Sample data loaded!")
                st.rerun()
            
            if st.session_state.df is not None and st.button("🔍 Quick Analysis", use_container_width=True):
                st.session_state.current_page = "🔍 Data Analysis"
                st.rerun()
            
            if st.session_state.model_trained and st.button("🎯 Make Predictions", use_container_width=True):
                st.session_state.current_page = "🎯 Predictions"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Status Overview
        st.markdown('<h2 class="section-header">📊 Project Status</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "✅ Loaded" if st.session_state.data_loaded else "⚠️ Required"
            color = "#64ffda" if st.session_state.data_loaded else "#FF9800"
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Data Status</div>
                <div class="metric-value" style="color:{color}">{status}</div>
                <div style="font-size: 0.9rem; color: #a8b2d1;">
                    {len(st.session_state.df) if st.session_state.df is not None else 0} samples
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            status = "✅ Complete" if st.session_state.analysis_complete else "🔄 Pending"
            color = "#64ffda" if st.session_state.analysis_complete else "#a8b2d1"
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Analysis</div>
                <div class="metric-value" style="color:{color}">{status}</div>
                <div style="font-size: 0.9rem; color: #a8b2d1;">
                    {len(st.session_state.insights) if st.session_state.insights else 0} insights
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            status = "✅ Complete" if st.session_state.preprocessing_complete else "🔄 Pending"
            color = "#64ffda" if st.session_state.preprocessing_complete else "#a8b2d1"
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Preprocessing</div>
                <div class="metric-value" style="color:{color}">{status}</div>
                <div style="font-size: 0.9rem; color: #a8b2d1;">
                    Ready for modeling
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            status = "✅ Trained" if st.session_state.model_trained else "⚠️ Required"
            color = "#64ffda" if st.session_state.model_trained else "#FF9800"
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Model Status</div>
                <div class="metric-value" style="color:{color}">{status}</div>
                <div style="font-size: 0.9rem; color: #a8b2d1;">
                    {st.session_state.model.__class__.__name__ if st.session_state.model else "No model"}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Recent Activity
        if st.session_state.training_history:
            st.markdown('<h2 class="section-header">📈 Recent Activity</h2>', unsafe_allow_html=True)
            
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            for history in st.session_state.training_history[-3:]:  # Show last 3
                st.markdown(f"""
                <div class="insight-box">
                    <div class="insight-title">🔄 {history.get('model_name', 'Training')}</div>
                    <div class="insight-content">
                        <strong>Time:</strong> {history.get('timestamp').strftime('%Y-%m-%d %H:%M')}<br>
                        <strong>Accuracy:</strong> {history.get('accuracy', 'N/A'):.2%}<br>
                        <strong>Features:</strong> {len(history.get('features', []))}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_data_loader(self):
        """Render data loading interface"""
        st.markdown('<h1 class="main-header">📁 Data Loader</h1>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-header">📤 Upload Your Data</h2>', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose a CSV or Excel file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload your dataset for analysis"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.session_state.current_step = 2
                    
                    st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                    
                    # Show preview
                    with st.expander("📋 Data Preview", expanded=True):
                        st.dataframe(df.head(), use_container_width=True)
                        
                    # Basic info
                    with st.expander("📊 Basic Information", expanded=True):
                        col_info1, col_info2, col_info3 = st.columns(3)
                        with col_info1:
                            st.metric("Rows", len(df))
                        with col_info2:
                            st.metric("Columns", len(df.columns))
                        with col_info3:
                            st.metric("Missing Values", df.isnull().sum().sum())
                
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">💡 Quick Load</h3>', unsafe_allow_html=True)
            
            if st.button("📊 Load Sample Dataset", use_container_width=True):
                with st.spinner("Loading sample data..."):
                    self.load_sample_data()
                    st.success("Sample data loaded!")
                    st.rerun()
            
            st.markdown("---")
            st.markdown("""
            <div class="insight-content">
            <strong>Supported formats:</strong>
            <ul>
                <li>CSV (.csv)</li>
                <li>Excel (.xlsx, .xls)</li>
            </ul>
            <strong>Requirements:</strong>
            <ul>
                <li>First row as headers</li>
                <li>Clean column names</li>
                <li>Target column should be identified</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.df is not None:
            # Data information using YOUR class
            st.markdown('<h2 class="section-header">📋 Data Information (Using YOUR Class)</h2>', unsafe_allow_html=True)
            
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            
            with st.spinner("Analyzing data with YOUR logic..."):
                info_df = self.ml.data_information(st.session_state.df)
                
                tab1, tab2, tab3 = st.tabs(["📊 Complete Info", "📈 Statistics", "⚠️ Issues"])
                
                with tab1:
                    # Convert object columns to string for display
                    info_df_display = info_df.copy()
                    info_df_display['Values'] = info_df_display['Values'].astype(str)

                    # Display the dataframe
                    st.dataframe(
                        info_df_display,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Add insight
                    null_cols = info_df[info_df['Null Num'] > 0]['Names'].tolist()
                    if null_cols:
                        self.add_insight(
                            "Null Values Detected",
                            f"Found null values in columns: {', '.join(null_cols)}. Consider handling these in preprocessing."
                        )
                
                with tab2:
                    st.dataframe(
                        st.session_state.df.describe().T.style.bar(
                            subset=['mean'], 
                            color='#FFA07A'
                        ).background_gradient(
                            subset=['std', '50%', 'max'], 
                            cmap='Blues'
                        ),
                        use_container_width=True,
                        height=400
                    )
                
                with tab3:
                    issues = []
                    
                    # Check for nulls
                    null_count = st.session_state.df.isnull().sum().sum()
                    if null_count > 0:
                        issues.append(f"❌ {null_count} missing values found")
                    
                    # Check for duplicates
                    dup_count = st.session_state.df.duplicated().sum()
                    if dup_count > 0:
                        issues.append(f"❌ {dup_count} duplicate rows found")
                    
                    # Check data types
                    object_cols = st.session_state.df.select_dtypes(include=['object']).columns.tolist()
                    if object_cols:
                        issues.append(f"⚠️ {len(object_cols)} categorical columns need encoding")
                    
                    if issues:
                        for issue in issues:
                            st.warning(issue)
                    else:
                        st.success("✅ No major issues found!")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def load_sample_data(self):
        """Load sample data for demonstration"""
        # Create sample data similar to your example
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = {
            'CustomerID': range(1, n_samples + 1),
            'Churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'Tenure': np.random.randint(1, 60, n_samples),
            'PreferredLoginDevice': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_samples),
            'CityTier': np.random.choice([1, 2, 3], n_samples),
            'WarehouseToHome': np.random.randint(1, 30, n_samples),
            'PreferredPaymentMode': np.random.choice(['Credit Card', 'Debit Card', 'UPI', 'Cash'], n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'HourSpendOnApp': np.random.randint(1, 10, n_samples),
            'NumberOfDeviceRegistered': np.random.randint(1, 5, n_samples),
            'PreferedOrderCat': np.random.choice(['Electronics', 'Clothing', 'Groceries', 'Books'], n_samples),
            'SatisfactionScore': np.random.randint(1, 6, n_samples),
            'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
            'NumberOfAddress': np.random.randint(1, 10, n_samples),
            'Complain': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'OrderAmountHikeFromlastYear': np.random.randint(10, 50, n_samples),
            'CouponUsed': np.random.randint(0, 10, n_samples),
            'OrderCount': np.random.randint(1, 50, n_samples),
            'DaySinceLastOrder': np.random.randint(1, 30, n_samples),
            'CashbackAmount': np.random.uniform(10, 500, n_samples).round(2)
        }
        
        df = pd.DataFrame(sample_data)
        
        # Add some nulls for demonstration
        for col in ['Tenure', 'WarehouseToHome', 'HourSpendOnApp']:
            idx = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
            df.loc[idx, col] = np.nan
        
        st.session_state.df = df
        st.session_state.data_loaded = True
        st.session_state.current_step = 2
    
    def render_data_analysis(self):
        """Render comprehensive data analysis"""
        if not st.session_state.data_loaded:
            st.warning("Please load data first!")
            return
        
        st.markdown('<h1 class="main-header">🔍 Data Analysis</h1>', unsafe_allow_html=True)
        
        # Analysis controls
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">⚙️ Analysis Controls</h3>', unsafe_allow_html=True)
            
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Target Analysis", "Numerical Analysis", "Categorical Analysis", 
                 "Correlation Analysis", "Custom Visualization"]
            )
            
            if st.button("🔄 Run Complete Analysis", use_container_width=True):
                with st.spinner("Running comprehensive analysis..."):
                    self.run_complete_analysis()
                    st.session_state.analysis_complete = True
                    st.session_state.current_step = 3
                    st.success("Analysis complete!")
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display analysis based on type
        if analysis_type == "Target Analysis":
            self.render_target_analysis()
        elif analysis_type == "Numerical Analysis":
            self.render_numerical_analysis()
        elif analysis_type == "Categorical Analysis":
            self.render_categorical_analysis()
        elif analysis_type == "Correlation Analysis":
            self.render_correlation_analysis()
        else:
            self.render_custom_visualization()
    
    def run_complete_analysis(self):
        """Run complete analysis and generate insights"""
        df = st.session_state.df
        
        # Target distribution insight
        if 'Churn' in df.columns:
            churn_rate = df['Churn'].mean()
            self.add_insight(
                "Target Distribution",
                f"Churn rate is {churn_rate:.1%}. {'Class imbalance detected!' if churn_rate < 0.3 or churn_rate > 0.7 else 'Balanced dataset.'}"
            )
        
        # Missing values insight
        null_cols = df.columns[df.isnull().any()].tolist()
        if null_cols:
            self.add_insight(
                "Missing Values",
                f"Found missing values in {len(null_cols)} columns: {', '.join(null_cols[:5])}{'...' if len(null_cols) > 5 else ''}"
            )
        
        # Data types insight
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.add_insight(
            "Data Types",
            f"Dataset has {len(numeric_cols)} numerical and {len(cat_cols)} categorical columns"
        )
    
    def render_target_analysis(self):
        """Render target variable analysis"""
        df = st.session_state.df
        
        # Find target column
        target_candidates = ['Churn', 'target', 'label', 'Class']
        target_col = None
        for col in target_candidates:
            if col in df.columns:
                target_col = col
                break
        
        if target_col:
            st.markdown(f'<h2 class="section-header">🎯 Target Analysis: {target_col}</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                # Using YOUR pie_chart method
                fig = plt.figure(figsize=(8, 6))
                self.ml.pie_chart(target_col, df)
                st.pyplot(fig)
                self.save_plot(fig, f"{target_col} Distribution")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                # Using Plotly for interactive plot
                target_counts = df[target_col].value_counts()
                fig = px.bar(
                    x=target_counts.index.astype(str),
                    y=target_counts.values,
                    title=f"{target_col} Distribution",
                    color=target_counts.index.astype(str),
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig, use_container_width=True)
                self.save_plot(fig, f"{target_col} Bar Chart")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Insights column
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">💡 Target Insights</h3>', unsafe_allow_html=True)
            
            target_stats = df[target_col].value_counts(normalize=True)
            insight_text = f"""
            <div class="insight-content">
            <p><strong>Distribution:</strong> Class 0: {target_stats.get(0, 0):.1%}, Class 1: {target_stats.get(1, 0):.1%}</p>
            <p><strong>Imbalance:</strong> {'⚠️ Class imbalance detected!' if abs(target_stats.get(0, 0) - target_stats.get(1, 0)) > 0.3 else '✅ Balanced dataset'}</p>
            <p><strong>Recommendation:</strong> Consider {'class balancing techniques' if abs(target_stats.get(0, 0) - target_stats.get(1, 0)) > 0.3 else 'proceeding with current distribution'}</p>
            </div>
            """
            st.markdown(insight_text, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No target column found. Looking for columns named: 'Churn', 'target', 'label', or 'Class'")
    
    def render_numerical_analysis(self):
        """Render numerical column analysis"""
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numerical columns found!")
            return
        
        st.markdown('<h2 class="section-header">📈 Numerical Analysis</h2>', unsafe_allow_html=True)
        
        # Column selector
        selected_col = st.selectbox("Select Numerical Column", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            # Using YOUR histogram_plot method
            fig = plt.figure(figsize=(10, 6))
            self.ml.histogram_plot(selected_col, df)
            st.pyplot(fig)
            self.save_plot(fig, f"{selected_col} Histogram")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            # Using YOUR box_plot method if target exists
            target_candidates = ['Churn', 'target', 'label', 'Class']
            target_col = None
            for col in target_candidates:
                if col in df.columns:
                    target_col = col
                    break
            
            if target_col:
                fig = plt.figure(figsize=(10, 6))
                self.ml.box_plot(target_col, selected_col, df)
                st.pyplot(fig)
                self.save_plot(fig, f"{selected_col} by {target_col} Box Plot")
            else:
                # Regular box plot
                fig = plt.figure(figsize=(10, 6))
                plt.boxplot(df[selected_col].dropna())
                plt.title(f"Box Plot of {selected_col}")
                plt.xlabel(selected_col)
                st.pyplot(fig)
                self.save_plot(fig, f"{selected_col} Box Plot")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Statistics and insights
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Mean", f"{df[selected_col].mean():.2f}")
        with col_stat2:
            st.metric("Std Dev", f"{df[selected_col].std():.2f}")
        with col_stat3:
            st.metric("Skewness", f"{skew(df[selected_col].dropna()):.2f}")
        with col_stat4:
            outlier_count = len(self.detect_outliers(df[selected_col]))
            st.metric("Outliers", outlier_count)
        
        # Add insight based on statistics
        skew_val = skew(df[selected_col].dropna())
        if abs(skew_val) > 1:
            skew_type = "highly skewed" if abs(skew_val) > 1 else "moderately skewed"
            self.add_insight(
                f"{selected_col} Distribution",
                f"Column is {skew_type} (skewness: {skew_val:.2f}). Consider transformation."
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_categorical_analysis(self):
        """Render categorical column analysis"""
        df = st.session_state.df
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if not cat_cols:
            st.warning("No categorical columns found!")
            return
        
        st.markdown('<h2 class="section-header">📊 Categorical Analysis</h2>', unsafe_allow_html=True)
        
        # Column selector
        selected_col = st.selectbox("Select Categorical Column", cat_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            # Using YOUR bar_plot method
            fig = plt.figure(figsize=(10, 6))
            self.ml.bar_plot(selected_col, df)
            st.pyplot(fig)
            self.save_plot(fig, f"{selected_col} Bar Plot")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            # Interactive count plot with Plotly
            value_counts = df[selected_col].value_counts()
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"{selected_col} Distribution",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
            self.save_plot(fig, f"{selected_col} Pie Chart")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Statistics and insights
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        
        unique_values = df[selected_col].nunique()
        most_common = df[selected_col].mode()[0] if not df[selected_col].mode().empty else "N/A"
        most_common_pct = (df[selected_col] == most_common).mean() * 100
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Unique Values", unique_values)
        with col_stat2:
            st.metric("Most Common", most_common)
        with col_stat3:
            st.metric("Frequency", f"{most_common_pct:.1f}%")
        
        # Encoding recommendation
        st.markdown('<h3 class="sub-header">🔧 Encoding Recommendation</h3>', unsafe_allow_html=True)
        
        if unique_values <= 5:
            encoding_rec = "One-Hot Encoding (small number of categories)"
        elif unique_values <= 20:
            encoding_rec = "Ordinal Encoding if ordinal, else One-Hot"
        else:
            encoding_rec = "Target Encoding or Frequency Encoding (many categories)"
        
        st.info(f"**Recommended encoding:** {encoding_rec}")
        
        # Add insight
        self.add_insight(
            f"{selected_col} Analysis",
            f"Column has {unique_values} unique values. {encoding_rec}"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_correlation_analysis(self):
        """Render correlation analysis"""
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numerical columns for correlation analysis!")
            return
        
        st.markdown('<h2 class="section-header">📐 Correlation Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">⚙️ Settings</h3>', unsafe_allow_html=True)
            
            # Select columns for correlation
            selected_cols = st.multiselect(
                "Select columns for correlation",
                numeric_cols,
                default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
            )
            
            correlation_method = st.selectbox(
                "Correlation Method",
                ["Pearson", "Spearman", "Kendall"]
            )
            
            if st.button("🔄 Calculate Correlation", use_container_width=True):
                with st.spinner("Calculating correlations..."):
                    pass  # Calculation happens in display
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if selected_cols:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                
                # Calculate correlation matrix
                corr_matrix = df[selected_cols].corr(method=correlation_method.lower())
                
                # Plot heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    center=0,
                    square=True,
                    ax=ax
                )
                plt.title(f"{correlation_method} Correlation Matrix")
                st.pyplot(fig)
                self.save_plot(fig, f"{correlation_method} Correlation Heatmap")
                
                # Find strong correlations
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            strong_corr.append(
                                f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.2f}"
                            )
                
                if strong_corr:
                    self.add_insight(
                        "Strong Correlations",
                        f"Found {len(strong_corr)} strong correlations (>0.7): {', '.join(strong_corr[:3])}"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Correlation insights
        if selected_cols:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">💡 Correlation Insights</h3>', unsafe_allow_html=True)
            
            # Calculate top correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'pair': f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}",
                        'value': corr_matrix.iloc[i, j]
                    })
            
            # Sort by absolute correlation
            corr_pairs.sort(key=lambda x: abs(x['value']), reverse=True)
            
            st.markdown("**Top Correlations:**")
            for pair in corr_pairs[:5]:
                color = "🟢" if abs(pair['value']) < 0.3 else "🟡" if abs(pair['value']) < 0.7 else "🔴"
                st.markdown(f"{color} **{pair['pair']}**: {pair['value']:.3f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_custom_visualization(self):
        """Render custom visualization interface"""
        df = st.session_state.df
        
        st.markdown('<h2 class="section-header">🎨 Custom Visualization</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">⚙️ Plot Settings</h3>', unsafe_allow_html=True)
            
            # Plot type selection
            plot_type = st.selectbox(
                "Select Plot Type",
                ["Scatter Plot", "Line Plot", "Violin Plot", "Strip Plot", 
                 "KDE Plot", "Heatmap", "Count Plot", "Box Plot"]
            )
            
            # Column selection based on plot type
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if plot_type in ["Scatter Plot", "Line Plot"]:
                x_col = st.selectbox("X-axis", numeric_cols)
                y_col = st.selectbox("Y-axis", numeric_cols)
                hue_col = st.selectbox("Hue (optional)", [None] + cat_cols)
                
                if st.button("📊 Generate Plot", use_container_width=True):
                    with st.spinner("Creating plot..."):
                        if plot_type == "Scatter Plot":
                            fig = plt.figure(figsize=(10, 6))
                            self.ml.scatter_plot(x_col, y_col, hue_col, df)
                            st.session_state.last_plot = fig
                        else:
                            fig = plt.figure(figsize=(10, 6))
                            self.ml.line_plot(x_col, y_col, hue_col, df)
                            st.session_state.last_plot = fig
            
            elif plot_type in ["Violin Plot", "Strip Plot", "Box Plot"]:
                x_col = st.selectbox("X-axis", cat_cols + numeric_cols[:5])
                y_col = st.selectbox("Y-axis", numeric_cols)
                hue_col = st.selectbox("Hue (optional)", [None] + cat_cols)
                
                if st.button("📊 Generate Plot", use_container_width=True):
                    with st.spinner("Creating plot..."):
                        if plot_type == "Violin Plot":
                            fig = plt.figure(figsize=(10, 6))
                            self.ml.violin_plot(x_col, y_col, hue_col, df)
                        elif plot_type == "Strip Plot":
                            fig = plt.figure(figsize=(10, 6))
                            self.ml.strip_plot(x_col, y_col, df)
                        else:
                            fig = plt.figure(figsize=(10, 6))
                            self.ml.box_plot(x_col, y_col, df)
                        st.session_state.last_plot = fig
            
            elif plot_type == "KDE Plot":
                col = st.selectbox("Column", numeric_cols)
                hue_col = st.selectbox("Hue", [None] + cat_cols)
                
                if st.button("📊 Generate Plot", use_container_width=True):
                    with st.spinner("Creating plot..."):
                        fig = plt.figure(figsize=(10, 6))
                        self.ml.kde_plot(col, hue_col, df)
                        st.session_state.last_plot = fig
            
            elif plot_type == "Count Plot":
                col = st.selectbox("Column", cat_cols)
                hue_col = st.selectbox("Hue (optional)", [None] + cat_cols)
                
                if st.button("📊 Generate Plot", use_container_width=True):
                    with st.spinner("Creating plot..."):
                        fig = plt.figure(figsize=(10, 6))
                        self.ml.count_plot(col, hue_col, df)
                        st.session_state.last_plot = fig
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if hasattr(st.session_state, 'last_plot'):
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.pyplot(st.session_state.last_plot)
                
                # Save plot button
                if st.button("💾 Save Plot", use_container_width=True):
                    plot_name = f"{plot_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.save_plot(st.session_state.last_plot, plot_name)
                    st.success(f"Plot saved as {plot_name}")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    def render_preprocessing(self):
        """Render preprocessing interface"""
        if not st.session_state.data_loaded:
            st.warning("Please load data first!")
            return
        
        st.markdown('<h1 class="main-header">⚙️ Preprocessing</h1>', unsafe_allow_html=True)
        
        # Preprocessing steps
        steps = [
            ("Handle Missing Values", self.render_missing_values),
            ("Handle Outliers", self.render_outliers),
            ("Feature Engineering", self.render_feature_engineering),
            ("Encoding", self.render_encoding),
            ("Scaling", self.render_scaling)
        ]
        
        # Create tabs for each step
        tabs = st.tabs([step[0] for step in steps])
        
        for i, (step_name, step_func) in enumerate(steps):
            with tabs[i]:
                step_func()
        
        # Apply all preprocessing
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">🚀 Apply All Preprocessing</h3>', unsafe_allow_html=True)
        
        if st.button("🔄 Apply All Preprocessing Steps", use_container_width=True, type="primary"):
            with st.spinner("Applying preprocessing pipeline..."):
                self.apply_preprocessing_pipeline()
                st.session_state.preprocessing_complete = True
                st.session_state.current_step = 4
                st.success("Preprocessing complete! Data ready for modeling.")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_missing_values(self):
        """Render missing values handling interface"""
        df = st.session_state.df
        
        # Calculate missing values
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0]
        
        st.markdown('<h3 class="sub-header">🔍 Missing Values Analysis</h3>', unsafe_allow_html=True)
        
        if len(missing_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(
                    missing_df.sort_values('Missing_Percentage', ascending=False),
                    use_container_width=True
                )
            
            with col2:
                # Visualization
                fig = px.bar(
                    missing_df,
                    x='Column',
                    y='Missing_Percentage',
                    title='Missing Values Percentage',
                    color='Missing_Percentage',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Handling options
            st.markdown('<h3 class="sub-header">⚙️ Handling Strategy</h3>', unsafe_allow_html=True)
            
            selected_cols = st.multiselect(
                "Select columns to handle",
                missing_df['Column'].tolist(),
                default=missing_df['Column'].tolist()[:3]
            )
            
            handling_method = st.selectbox(
                "Select handling method",
                ["Mean", "Median", "Mode", "KNN Imputer", "Drop Rows", "Drop Column"]
            )
            
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                if st.button("🔧 Apply to Selected Columns", use_container_width=True):
                    with st.spinner(f"Applying {handling_method}..."):
                        self.handle_missing_values(selected_cols, handling_method)
                        st.success(f"Applied {handling_method} to {len(selected_cols)} columns")
                        st.rerun()
            
            with col_opt2:
                if st.button("🔧 Apply Smart Strategy", use_container_width=True):
                    with st.spinner("Applying smart strategy..."):
                        self.apply_smart_missing_handling()
                        st.success("Smart strategy applied")
                        st.rerun()
        else:
            st.success("✅ No missing values found!")
    
    def handle_missing_values(self, columns, method):
        """Handle missing values using YOUR class"""
        if method == "Drop Rows":
            st.session_state.df.dropna(subset=columns, inplace=True)
        elif method == "Drop Column":
            st.session_state.df.drop(columns=columns, inplace=True)
        else:
            # Map method names to YOUR class method
            method_map = {
                "Mean": "median",  # Using median instead of mean
                "Median": "median",
                "Mode": "mode",
                "KNN Imputer": "knn imputer"
            }
            self.ml.handle_null_values(method_map[method], columns, st.session_state.df)
    
    def apply_smart_missing_handling(self):
        """Apply smart missing value handling strategy"""
        df = st.session_state.df
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    # Use median for numerical with outliers
                    self.ml.handle_null_values('median', [col], df)
                else:
                    # Use mode for categorical
                    self.ml.handle_null_values('mode', [col], df)
    
    def render_outliers(self):
        """Render outliers handling interface"""
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        st.markdown('<h3 class="sub-header">🔍 Outlier Detection</h3>', unsafe_allow_html=True)
        
        if numeric_cols:
            selected_cols = st.multiselect(
                "Select columns for outlier analysis",
                numeric_cols,
                default=numeric_cols[:3]
            )
            
            if selected_cols:
                # Show box plots using YOUR class
                fig = self.ml.check_outliers(selected_cols, df)
                st.pyplot(fig)
                
                # Outlier statistics
                st.markdown('<h3 class="sub-header">📊 Outlier Statistics</h3>', unsafe_allow_html=True)
                
                outlier_stats = []
                for col in selected_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    outlier_stats.append({
                        'Column': col,
                        'Lower Bound': lower_bound,
                        'Upper Bound': upper_bound,
                        'Outlier Count': len(outliers),
                        'Outlier Percentage': (len(outliers) / len(df) * 100)
                    })
                
                outlier_df = pd.DataFrame(outlier_stats)
                st.dataframe(outlier_df, use_container_width=True)
                
                # Handling options
                st.markdown('<h3 class="sub-header">⚙️ Outlier Handling</h3>', unsafe_allow_html=True)
                
                handle_method = st.selectbox(
                    "Select handling method",
                    ["Cap at bounds", "Remove outliers", "Transform", "Keep as is"]
                )
                
                if st.button("🔧 Handle Outliers", use_container_width=True):
                    with st.spinner("Handling outliers..."):
                        if handle_method == "Cap at bounds":
                            for col in selected_cols:
                                Q1 = df[col].quantile(0.25)
                                Q3 = df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                
                                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
                        
                        elif handle_method == "Remove outliers":
                            for col in selected_cols:
                                Q1 = df[col].quantile(0.25)
                                Q3 = df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                
                                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                        
                        st.session_state.df = df
                        st.success(f"Outliers handled using {handle_method}")
                        st.rerun()
        else:
            st.info("No numerical columns for outlier analysis")
    
    def render_feature_engineering(self):
        """Render feature engineering interface"""
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        st.markdown('<h3 class="sub-header">🔧 Feature Engineering</h3>', unsafe_allow_html=True)
        
        # Simple feature engineering options
        engineering_options = st.multiselect(
            "Select engineering operations",
            [
                "Create Interaction Terms",
                "Polynomial Features",
                "Log Transformation",
                "Binning",
                "Date Features (if available)"
            ]
        )
        
        if "Create Interaction Terms" in engineering_options:
            col1, col2 = st.columns(2)
            with col1:
                interact_col1 = st.selectbox("First column", numeric_cols)
            with col2:
                interact_col2 = st.selectbox("Second column", numeric_cols)
            
            if st.button("➕ Create Interaction", use_container_width=True):
                new_col_name = f"{interact_col1}_x_{interact_col2}"
                df[new_col_name] = df[interact_col1] * df[interact_col2]
                st.success(f"Created interaction feature: {new_col_name}")
        
        if "Log Transformation" in engineering_options:
            log_col = st.selectbox("Select column for log transform", numeric_cols)
            if st.button("📐 Apply Log Transform", use_container_width=True):
                df[f"log_{log_col}"] = np.log1p(df[log_col])
                st.success(f"Created log transformation: log_{log_col}")
        
        if "Binning" in engineering_options:
            bin_col = st.selectbox("Select column for binning", numeric_cols)
            n_bins = st.slider("Number of bins", 3, 10, 5)
            if st.button("📦 Create Bins", use_container_width=True):
                df[f"{bin_col}_bins"] = pd.cut(df[bin_col], bins=n_bins, labels=False)
                st.success(f"Created {n_bins} bins for {bin_col}")
        
        st.session_state.df = df
        
        # Show engineered features
        if len(df.columns) > len(numeric_cols):  # If new features were added
            st.markdown('<h3 class="sub-header">📋 Engineered Features</h3>', unsafe_allow_html=True)
            new_features = [col for col in df.columns if col not in numeric_cols and df[col].dtype in ['int64', 'float64']]
            if new_features:
                st.write(f"New features: {', '.join(new_features)}")
    
    def render_encoding(self):
        """Render encoding interface"""
        df = st.session_state.df
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        st.markdown('<h3 class="sub-header">🔤 Encoding Categorical Variables</h3>', unsafe_allow_html=True)
        
        if cat_cols:
            # Show categorical columns
            st.write(f"**Categorical columns found:** {', '.join(cat_cols)}")
            
            # Encoding strategy for each column
            encoding_strategies = {}
            for col in cat_cols:
                unique_count = df[col].nunique()
                st.markdown(f"**{col}** ({unique_count} unique values)")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    if unique_count <= 10:
                        default_method = "One-Hot"
                    elif unique_count <= 20:
                        default_method = "Ordinal"
                    else:
                        default_method = "Target"
                    
                    method = st.selectbox(
                        f"Method for {col}",
                        ["One-Hot", "Ordinal", "Target", "Frequency", "Binary", "Leave as is"],
                        index=["One-Hot", "Ordinal", "Target", "Frequency", "Binary", "Leave as is"].index(default_method),
                        key=f"encode_{col}"
                    )
                
                with col2:
                    if st.button(f"Apply", key=f"btn_{col}", use_container_width=True):
                        encoding_strategies[col] = method
            
            # Apply all encodings
            if st.button("🔧 Apply All Encodings", use_container_width=True, type="primary"):
                with st.spinner("Applying encodings..."):
                    self.apply_encodings(encoding_strategies)
                    st.success("Encodings applied!")
                    st.rerun()
        else:
            st.success("✅ No categorical columns to encode!")
    
    def apply_encodings(self, strategies):
        """Apply encoding strategies"""
        df = st.session_state.df
        
        for col, method in strategies.items():
            if method == "One-Hot":
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            elif method == "Ordinal":
                encoder = OrdinalEncoder()
                df[col] = encoder.fit_transform(df[[col]])
            elif method == "Frequency":
                freq_map = df[col].value_counts().to_dict()
                df[col] = df[col].map(freq_map)
        
        st.session_state.df = df
    
    def render_scaling(self):
        """Render scaling interface"""
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        st.markdown('<h3 class="sub-header">⚖️ Feature Scaling</h3>', unsafe_allow_html=True)
        
        if numeric_cols:
            # Select columns to scale
            scale_cols = st.multiselect(
                "Select columns to scale",
                numeric_cols,
                default=numeric_cols[:5]
            )
            
            # Select scaling method
            scale_method = st.selectbox(
                "Scaling method",
                ["Standard Scaler", "Min-Max Scaler", "Robust Scaler", "No Scaling"]
            )
            
            # Show before/after comparison
            if scale_cols:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Before Scaling**")
                    st.dataframe(df[scale_cols].describe().round(2), use_container_width=True)
                
                # Apply scaling
                if st.button("⚖️ Apply Scaling", use_container_width=True):
                    with st.spinner("Applying scaling..."):
                        # Save original for comparison
                        st.session_state.df_original = df.copy()
                        
                        # Apply scaling
                        if scale_method == "Standard Scaler":
                            scaler = StandardScaler()
                            df[scale_cols] = scaler.fit_transform(df[scale_cols])
                        elif scale_method == "Min-Max Scaler":
                            scaler = MinMaxScaler()
                            df[scale_cols] = scaler.fit_transform(df[scale_cols])
                        elif scale_method == "Robust Scaler":
                            scaler = RobustScaler()
                            df[scale_cols] = scaler.fit_transform(df[scale_cols])
                        
                        st.session_state.df = df
                        
                        with col2:
                            st.markdown("**After Scaling**")
                            st.dataframe(df[scale_cols].describe().round(2), use_container_width=True)
                        
                        st.success(f"Applied {scale_method}")
        else:
            st.info("No numerical columns for scaling")
    
    def apply_preprocessing_pipeline(self):
        """Apply complete preprocessing pipeline"""
        df = st.session_state.df.copy()
        
        # Save processed dataframe
        st.session_state.df_processed = df
        
        # Add insight
        self.add_insight(
            "Preprocessing Complete",
            "All preprocessing steps applied. Data is ready for modeling."
        )
    
    def render_model_training(self):
        """Render model training interface"""
        if not st.session_state.data_loaded:
            st.warning("Please load and preprocess data first!")
            return
        
        st.markdown('<h1 class="main-header">🤖 Model Training</h1>', unsafe_allow_html=True)
        
        # Model selection and configuration
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">⚙️ Model Configuration</h3>', unsafe_allow_html=True)
            
            # Problem type
            problem_type = st.selectbox(
                "Problem Type",
                ["Classification", "Regression", "Clustering"]
            )
            
            # Target selection
            if problem_type != "Clustering":
                target_col = st.selectbox(
                    "Target Column",
                    st.session_state.df.columns.tolist()
                )
            
            # Model selection
            if problem_type == "Classification":
                model_type = st.selectbox(
                    "Model Algorithm",
                    ["Random Forest", "Logistic Regression", "XGBoost", 
                     "Support Vector Machine", "K-Nearest Neighbors", "Decision Tree"]
                )
            elif problem_type == "Regression":
                model_type = st.selectbox(
                    "Model Algorithm",
                    ["Random Forest", "Linear Regression", "XGBoost", 
                     "Support Vector Regression", "Decision Tree"]
                )
            else:
                model_type = st.selectbox(
                    "Clustering Algorithm",
                    ["K-Means", "DBSCAN", "Agglomerative"]
                )
            
            # Hyperparameter tuning
            with st.expander("⚡ Hyperparameters", expanded=False):
                if model_type == "Random Forest":
                    n_estimators = st.slider("n_estimators", 10, 500, 100, 10)
                    max_depth = st.slider("max_depth", 2, 50, 10, 2)
                    st.session_state.model_params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth
                    }
            
            # Train/test split
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            
            if st.button("🚀 Train Model", use_container_width=True, type="primary"):
                with st.spinner("Training model..."):
                    self.train_model(problem_type, model_type, target_col, test_size)
                    st.session_state.model_trained = True
                    st.session_state.current_step = 5
                    st.success("Model training complete!")
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if st.session_state.model_trained:
                st.markdown('<div class="content-card">', unsafe_allow_html=True)
                st.markdown('<h3 class="sub-header">📊 Training Results</h3>', unsafe_allow_html=True)
                
                # Show model performance
                if hasattr(st.session_state, 'model_performance'):
                    perf = st.session_state.model_performance
                    
                    col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
                    
                    if problem_type == "Classification":
                        with col_perf1:
                            st.metric("Accuracy", f"{perf.get('accuracy', 0):.2%}")
                        with col_perf2:
                            st.metric("Precision", f"{perf.get('precision', 0):.2%}")
                        with col_perf3:
                            st.metric("Recall", f"{perf.get('recall', 0):.2%}")
                        with col_perf4:
                            st.metric("F1-Score", f"{perf.get('f1', 0):.2%}")
                        
                        # Confusion matrix
                        if hasattr(st.session_state, 'confusion_matrix'):
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(
                                st.session_state.confusion_matrix,
                                annot=True,
                                fmt='d',
                                cmap='Blues',
                                ax=ax
                            )
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            ax.set_title('Confusion Matrix')
                            st.pyplot(fig)
                    
                    elif problem_type == "Regression":
                        with col_perf1:
                            st.metric("R² Score", f"{perf.get('r2', 0):.3f}")
                        with col_perf2:
                            st.metric("MAE", f"{perf.get('mae', 0):.3f}")
                        with col_perf3:
                            st.metric("MSE", f"{perf.get('mse', 0):.3f}")
                        with col_perf4:
                            st.metric("RMSE", f"{perf.get('rmse', 0):.3f}")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    def train_model(self, problem_type, model_type, target_col, test_size):
        """Train model using YOUR class methods"""
        df = st.session_state.df_processed if st.session_state.df_processed is not None else st.session_state.df
        
        try:
            # Split data using YOUR method
            x_train, x_test, y_train, y_test = self.ml.spliting_data(
                df, target_col, test_size=test_size
            )
            
            # Save split data
            st.session_state.x_train = x_train
            st.session_state.x_test = x_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            # Train model based on type
            if problem_type == "Classification":
                if model_type == "Random Forest":
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(
                        n_estimators=st.session_state.model_params.get('n_estimators', 100),
                        max_depth=st.session_state.model_params.get('max_depth', 10),
                        random_state=42
                    )
                elif model_type == "Logistic Regression":
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(random_state=42)
                elif model_type == "XGBoost":
                    from xgboost import XGBClassifier
                    model = XGBClassifier(random_state=42)
                elif model_type == "Support Vector Machine":
                    from sklearn.svm import SVC
                    model = SVC(random_state=42, probability=True)
                elif model_type == "K-Nearest Neighbors":
                    from sklearn.neighbors import KNeighborsClassifier
                    model = KNeighborsClassifier()
                else:  # Decision Tree
                    from sklearn.tree import DecisionTreeClassifier
                    model = DecisionTreeClassifier(random_state=42)
                
                # Train model
                model.fit(x_train, y_train)
                
                # Evaluate
                y_pred = model.predict(x_test)
                y_pred_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                
                st.session_state.model_performance = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc_score
                }
                
                # Confusion matrix
                st.session_state.confusion_matrix = confusion_matrix(y_test, y_pred)
                
                # Classification report
                st.session_state.classification_report = classification_report(y_test, y_pred, output_dict=True)
            
            elif problem_type == "Regression":
                if model_type == "Random Forest":
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(random_state=42)
                elif model_type == "Linear Regression":
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                elif model_type == "XGBoost":
                    from xgboost import XGBRegressor
                    model = XGBRegressor(random_state=42)
                elif model_type == "Support Vector Regression":
                    from sklearn.svm import SVR
                    model = SVR()
                else:  # Decision Tree
                    from sklearn.tree import DecisionTreeRegressor
                    model = DecisionTreeRegressor(random_state=42)
                
                # Train model
                model.fit(x_train, y_train)
                
                # Evaluate
                y_pred = model.predict(x_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                st.session_state.model_performance = {
                    'r2': r2,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse
                }
            
            else:  # Clustering
                if model_type == "K-Means":
                    from sklearn.cluster import KMeans
                    n_clusters = st.slider("Number of clusters", 2, 10, 3)
                    model = KMeans(n_clusters=n_clusters, random_state=42)
                elif model_type == "DBSCAN":
                    from sklearn.cluster import DBSCAN
                    model = DBSCAN()
                else:  # Agglomerative
                    from sklearn.cluster import AgglomerativeClustering
                    model = AgglomerativeClustering()
                
                # Fit model
                clusters = model.fit_predict(x_train)
                
                # Calculate metrics
                silhouette = silhouette_score(x_train, clusters)
                db_index = davies_bouldin_score(x_train, clusters)
                ch_score = calinski_harabasz_score(x_train, clusters)
                
                st.session_state.model_performance = {
                    'silhouette': silhouette,
                    'davies_bouldin': db_index,
                    'calinski_harabasz': ch_score
                }
            
            # Save model
            st.session_state.model = model
            st.session_state.feature_names = x_train.columns.tolist()
            
            # Add to training history
            st.session_state.training_history.append({
                'timestamp': datetime.now(),
                'model_name': f"{model_type} ({problem_type})",
                'problem_type': problem_type,
                'accuracy': accuracy if problem_type == "Classification" else r2 if problem_type == "Regression" else silhouette,
                'features': x_train.columns.tolist()
            })
            
            # Add insight
            self.add_insight(
                f"Model Training Complete: {model_type}",
                f"Trained {model_type} model for {problem_type} with {len(x_train.columns)} features."
            )
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
    
    def render_predictions(self):
        """Render prediction interface"""
        if not st.session_state.model_trained:
            st.warning("Please train a model first!")
            return
        
        st.markdown('<h1 class="main-header">🎯 Predictions</h1>', unsafe_allow_html=True)
        
        # Prediction interface
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">⚙️ Prediction Settings</h3>', unsafe_allow_html=True)
            
            prediction_type = st.radio(
                "Prediction Type",
                ["Single Prediction", "Batch Prediction", "Test Set Evaluation"]
            )
            
            if prediction_type == "Single Prediction":
                # Create input form based on feature names
                input_data = {}
                for feature in st.session_state.feature_names[:5]:  # Show first 5 features
                    input_data[feature] = st.number_input(
                        feature,
                        value=0.0,
                        step=0.1
                    )
                
                if st.button("🔮 Predict", use_container_width=True):
                    try:
                        # Prepare input
                        input_df = pd.DataFrame([input_data])
                        
                        # Make prediction
                        prediction = st.session_state.model.predict(input_df)
                        probability = st.session_state.model.predict_proba(input_df) if hasattr(st.session_state.model, "predict_proba") else None
                        
                        st.session_state.last_prediction = {
                            'input': input_data,
                            'prediction': prediction[0],
                            'probability': probability[0] if probability is not None else None
                        }
                        
                        st.success("Prediction complete!")
                        
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
            
            elif prediction_type == "Batch Prediction":
                uploaded_file = st.file_uploader(
                    "Upload prediction data",
                    type=['csv', 'xlsx']
                )
                
                if uploaded_file is not None:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            pred_df = pd.read_csv(uploaded_file)
                        else:
                            pred_df = pd.read_excel(uploaded_file)
                        
                        # Make predictions
                        predictions = st.session_state.model.predict(pred_df)
                        pred_df['Prediction'] = predictions
                        
                        st.session_state.batch_predictions = pred_df
                        
                        st.success(f"Made {len(predictions)} predictions!")
                        
                        # Download option
                        csv = pred_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error making predictions: {str(e)}")
            
            else:  # Test Set Evaluation
                if st.button("📊 Evaluate on Test Set", use_container_width=True):
                    try:
                        y_pred = st.session_state.model.predict(st.session_state.x_test)
                        
                        # Show metrics
                        if hasattr(st.session_state, 'model_performance'):
                            perf = st.session_state.model_performance
                            
                            st.markdown("**Test Set Performance:**")
                            for metric, value in perf.items():
                                st.metric(metric.replace('_', ' ').title(), 
                                         f"{value:.3f}")
                        
                    except Exception as e:
                        st.error(f"Evaluation error: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if hasattr(st.session_state, 'last_prediction'):
                st.markdown('<div class="content-card">', unsafe_allow_html=True)
                st.markdown('<h3 class="sub-header">📋 Prediction Result</h3>', unsafe_allow_html=True)
                
                pred = st.session_state.last_prediction
                
                # Display result
                st.markdown(f"""
                <div class="insight-box">
                    <div class="insight-title">Prediction Result</div>
                    <div class="insight-content">
                        <strong>Prediction:</strong> {pred['prediction']}<br>
                        {f"<strong>Probability:</strong> {pred['probability']}" if pred['probability'] is not None else ""}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show feature importance if available
                if hasattr(st.session_state.model, 'feature_importances_'):
                    st.markdown('<h3 class="sub-header">🔍 Feature Importance</h3>', unsafe_allow_html=True)
                    
                    importances = pd.DataFrame({
                        'Feature': st.session_state.feature_names,
                        'Importance': st.session_state.model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig = px.bar(
                        importances,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 10 Feature Importances',
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    def render_insights(self):
        """Render insights and explanations"""
        st.markdown('<h1 class="main-header">📈 Insights & Explanations</h1>', 
                   unsafe_allow_html=True)
        
        # Display all insights
        if st.session_state.insights:
            st.markdown('<h2 class="section-header">💡 Generated Insights</h2>', 
                       unsafe_allow_html=True)
            
            for insight in reversed(st.session_state.insights[-10:]):  # Show last 10
                st.markdown(f'''
                <div class="insight-box">
                    <div class="insight-title">{insight['title']}</div>
                    <div class="insight-content">
                        {insight['content']}<br>
                        <small style="color: #a8b2d1; font-size: 0.9rem;">
                            Step {insight['step']} • {insight['timestamp'].strftime('%Y-%m-%d %H:%M')}
                        </small>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("No insights generated yet. Complete the analysis steps to generate insights.")
        
        # Plot history
        if st.session_state.plot_history:
            st.markdown('<h2 class="section-header">📊 Plot History</h2>', 
                       unsafe_allow_html=True)
            
            for plot in reversed(st.session_state.plot_history[-5:]):  # Show last 5
                st.markdown(f'''
                <div class="content-card">
                    <h4>{plot['name']}</h4>
                    <p><small>Type: {plot['type']} • {plot['timestamp'].strftime('%Y-%m-%d %H:%M')}</small></p>
                </div>
                ''', unsafe_allow_html=True)
        
        # Model comparison if multiple models trained
        if len(st.session_state.training_history) > 1:
            st.markdown('<h2 class="section-header">🤖 Model Comparison</h2>', 
                       unsafe_allow_html=True)
            
            comparison_data = []
            for history in st.session_state.training_history:
                comparison_data.append({
                    'Model': history['model_name'],
                    'Accuracy': history.get('accuracy', 0),
                    'Timestamp': history['timestamp'],
                    'Features': len(history['features'])
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            fig = px.bar(
                comparison_df,
                x='Model',
                y='Accuracy',
                title='Model Performance Comparison',
                color='Accuracy',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def detect_outliers(self, series):
        """Detect outliers in a series"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series[(series < lower_bound) | (series > upper_bound)]
    
    def run(self):
        """Main application runner"""
        # Initialize current page if not set
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "🏠 Dashboard"
        
        # Create sidebar
        self.create_top_navigation()
        
        # Page routing
        pages = {
            "🏠 Dashboard": self.render_dashboard,
            "📁 Data Loader": self.render_data_loader,
            "🔍 Data Analysis": self.render_data_analysis,
            "⚙️ Preprocessing": self.render_preprocessing,
            "🤖 Model Training": self.render_model_training,
            "🎯 Predictions": self.render_predictions,
            "📈 Insights": self.render_insights
        }
        
        # Render current page
        if st.session_state.current_page in pages:
            pages[st.session_state.current_page]()

# Create machine_learning_class.py file
ml_class_code = '''
# machine_learning_class.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
from sklearn.feature_selection import f_classif, chi2, SelectKBest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    confusion_matrix, classification_report
)

class MachineLearning:
    """Your Machine Learning class implementation"""
    
    def data_information(self, data_frame):
        """
        Steps of understanding data to build the model ?
        
          Analysis :
            - plots of data frame for categorical & numerical
            - Some Math -> describe()
        
          Preprocessing :
            - Names of columns
            - Data types -> info() or here
            - nulls
            - duplicates
            - unique (num & values)
            - mean & median & mode : -> describe()
            - outliers -> box plot & math ( describe() )
        """
        
        """
        Just to know the information of each column.
        I get each name of it.
        Also each data type.
        """
        name_of_each_column = [col for col in data_frame]
        data_types_of_each_column = [data_frame[col].dtype for col in data_frame.columns]
        
        """
        Number of null values in each column is not enough to know if it is huge or not.
        I need to calculate the percentage of it based on length of data frame to make it clear.
        """
        null_values_of_each_column = [data_frame[col].isnull().sum() for col in data_frame.columns]
        percentage_of_null_values_of_each_column = [data_frame[col].isnull().sum() / len(data_frame) * 100 
                                                   for col in data_frame.columns]
        
        """
        Unique values make me know :
        
          - What are the exact values of each column ?
          - Column is categorical or numerical ?
        """
        num_of_unique_values_of_each_column = [data_frame[col].nunique() for col in data_frame.columns]
        unique_values_of_each_column = [data_frame[col].unique() for col in data_frame.columns]
        
        """
        What are the num of duplicates in data frame ?
        """
        duplicates = data_frame.duplicated().sum()
        
        information_of_data = pd.DataFrame(
            {
                'Names': name_of_each_column,
                'Values': unique_values_of_each_column,
                'Data Type': data_types_of_each_column,
                'Unique Num': num_of_unique_values_of_each_column,
                'Null Num': null_values_of_each_column,
                'Null Percentage': percentage_of_null_values_of_each_column,
                'Duplicates': duplicates
            }
        )
        
        return information_of_data
    
    def bar_plot(self, column, data_frame):
        sns.set_style("whitegrid")
        surviver_counts = data_frame[column].value_counts(normalize=True)*100
        
        plt.figure(figsize=(6, 6))
        ax = sns.barplot(x=surviver_counts.index, y=data_frame[column].value_counts(), palette="rocket")
        
        plt.title(f"Percentage of {column} and Non-{column}", fontsize=14, fontweight='bold')
        plt.xlabel(f"{column} Status", fontsize=12, fontweight='bold')
        plt.ylabel("Count", fontsize=12, fontweight='bold')
        
        for p, percentage in zip(ax.patches, surviver_counts.values):
            ax.annotate(f'{percentage:.1f}%',
                       (p.get_x() + p.get_width() / 2, p.get_height()),
                       ha='center', va='bottom',
                       fontsize=12, fontweight='bold', color='black')
        plt.show()
    
    def histogram_plot(self, column, data_frame):
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        
        skewness = skew(data_frame[column], nan_policy="omit")
        sns.histplot(data_frame[column], bins=50, kde=True, color="navy", edgecolor="black")
        
        plt.xlabel(column, fontsize=14, fontweight='bold')
        plt.ylabel("Frequency", fontsize=14, fontweight='bold')
        plt.title(f"Distribution of {column}", fontsize=16, fontweight='bold')
        
        plt.text(
            x=data_frame[column].max() * 0.7,
            y=plt.gca().get_ylim()[1] * 0.7,
            s=f"Skewness: {skewness:.2f}",
            fontsize=15, fontweight="bold", color="navy"
        )
        
        plt.show()
    
    def strip_plot(self, column_x, column_y, data_frame):
        plt.figure(figsize=(10, 6))
        sns.stripplot(x=data_frame[column_x], y=data_frame[column_y], jitter=True, alpha=0.7, 
                     palette=["#1f77b4", "#ff7f0e"])
        plt.xlabel(column_x, fontsize=14, fontweight="bold")
        plt.ylabel(column_y, fontsize=14, fontweight="bold")
        plt.title(f"Strip Plot of {column_y} by {column_x}", fontsize=16, fontweight="bold")
        plt.show()
    
    def pie_chart(self, column, data_frame):
        satisfaction_counts = data_frame[column].value_counts()
        
        plt.figure(figsize=(8, 6))
        plt.pie(satisfaction_counts,
                labels=satisfaction_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=plt.cm.Dark2.colors)
        
        plt.title(f"Distribution of {column}")
        plt.axis('equal')
        plt.show()
    
    def box_plot(self, column_x, column_y, data_frame):
        plt.figure(figsize=(10,5))
        sns.boxplot(x=column_x, y=column_y, data=data_frame, palette="rocket")
        
        plt.xlabel(column_x)
        plt.ylabel(column_y)
        plt.title(f'Box Plot of {column_y} by {column_x}')
        plt.show()
    
    def count_plot(self, column, hue, data_frame):
        ax = sns.countplot(x=column, data=data_frame, palette='rocket', hue=hue)
        
        for p in ax.patches:
            try:
                height = p.get_height()
                if height > 0:
                    ax.text(p.get_x() + p.get_width() / 2, height,
                           f'{height/len(data_frame)*100:.2f}%', 
                           ha='center', va='bottom', fontsize=10, color='black')
            except AttributeError:
                continue
        
        plt.title(f'Count of {column} with {hue} Status')
        plt.show()
    
    def scatter_plot(self, x, y, hue, data_frame):
        ax = sns.scatterplot(x=x, y=y, data=data_frame, hue=hue, palette='rocket')
        plt.title(f'Scatter plot of {x} vs {y} by {hue}')
        plt.show()
    
    def heatmap(self, data_frame):
        corr = data_frame.corr()
        ax = sns.heatmap(corr, annot=True, cmap="rocket", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()
    
    def violin_plot(self, x, y, hue, data_frame):
        ax = sns.violinplot(x=x, y=y, data=data_frame, hue=hue, palette='rocket', split=True)
        plt.title(f'Violin plot of {y} across {x} grouped by {hue}')
        plt.show()
    
    def swarm_plot(self, x, y, hue, data_frame):
        sns.swarmplot(x=x, y=y, data=data_frame, hue=hue, palette='rocket', dodge=True)
        plt.title(f'Swarm plot of {y} across {x} grouped by {hue}')
        plt.show()
    
    def kde_plot(self, column, hue, data_frame):
        sns.kdeplot(data=data_frame, x=column, hue=hue, fill=True, palette='rocket')
        plt.title(f'KDE plot of {column} by {hue}')
        plt.show()
    
    def line_plot(self, x, y, hue, data_frame):
        sns.lineplot(x=x, y=y, data=data_frame, hue=hue, palette='rocket')
        plt.title(f'Line plot of {y} vs {x} by {hue}')
        plt.show()
    
    def pixal_bar_plot(self, data_frame, x, label_x, label_y, y='Count'):
        counts = data_frame[x].value_counts().reset_index()
        counts.columns = [x, y]
        
        fig = px.bar(counts, 
                     x=x, 
                     y=y, 
                     title=f"Distribution of {x}",
                     labels={x: label_x, y: label_y},
                     color=y, 
                     color_continuous_scale="ylgnbu")
        
        fig.update_layout(
            title=dict(font=dict(size=24, family="Arial", color="black", weight="bold")),
            xaxis=dict(title_font=dict(size=20, family="Arial", color="black", weight="bold"), 
                       tickfont=dict(size=16, family="Arial", color="black")),
            yaxis=dict(title_font=dict(size=20, family="Arial", color="black", weight="bold"), 
                       tickfont=dict(size=16, family="Arial", color="black")),
            coloraxis_colorbar=dict(title_font=dict(size=16, family="Arial", color="black", weight="bold"))
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        fig.show()
    
    def another_bar_plot(self, data_frame, x, hue, labels_of_hue, y='Count'):
        counts = data_frame.groupby([x, hue]).size().unstack()
        counts.plot(kind='bar', stacked=True, figsize=(14, 6), color=['#1f77b4', '#ff7f0e'], edgecolor="black")
        
        plt.xlabel(x, fontsize=14, labelpad=10, fontweight="bold")
        plt.ylabel(y, fontsize=14, labelpad=10, fontweight="bold")
        plt.title(f"Distribution of {x}", fontsize=16, fontweight="bold")
        
        plt.legend(title=hue, labels=labels_of_hue)
        plt.show()
    
    def bar_pixel_plot(self, data_frame, x, hue):
        grouped = pd.DataFrame(data_frame.groupby(x)[hue].value_counts())
        grouped = grouped.rename(columns={hue: 'Count'})
        grouped = grouped.reset_index()
        
        fig = px.histogram(grouped, x=x, y='count',
                          color=hue, barmode='group', title="<b>"+f"{x} Vs {hue}", text_auto=True)
        
        fig.update_layout(
            title_font_color="black",
            template="plotly",
            title_font_size=30,
            title_x=0.5,
            xaxis_title=x,
            yaxis_title='count')
        
        fig.show()
    
    def for_each_type_bar_pixel_plot(self, data_frame, x, y):
        fig=px.histogram(data_frame, x=x, facet_col=y, color=y, text_auto=True,
                        title="<b>"+f"{y} Vs {x}")
        fig.update_layout(hovermode='x', title_font_size=30)
        
        fig.update_layout(
            title_font_color="black",
            template="plotly",
            title_font_size=30,
            hoverlabel_font_size=20,
            title_x=0.5,
            xaxis_title=x,
            yaxis_title='count')
        
        fig.show()
    
    def grouped_features_relation_with_target_pixel_plot(self, data_frame, x, y, hue):
        fig = px.histogram(
            data_frame,
            x=x,
            y=y,
            orientation="h",
            color=hue,
            text_auto=True,
            title=f"<b>{x} Vs {y}",
            color_discrete_sequence=['#BA1F33', '#3A506B', '#3E885B']
        )
        
        fig.update_layout(
            hovermode='x',
            title_font_size=30,
            title_font_color="black",
            template="plotly",
            title_x=0.5,
            xaxis_title=x,
            yaxis_title=y,
            hoverlabel_font_size=20,
            bargap=0.3
        )
        
        fig.show()
    
    def handle_null_values(self, handling_type, columns, data_frame):
        # handling_type -> mode , mean (not prefered) , median , knn imputer
        
        if handling_type == 'mode':
            for col in columns:
                # replace it with the most frequent value :
                data_frame[col] = data_frame[col].fillna(data_frame[col].mode()[0])
        
        elif handling_type == 'knn imputer':
            for col in columns:
                # replaces it with the previous value, and if it can't find it, then with the next one.
                data_frame[col] = data_frame[col].fillna(method='ffill').fillna(method='bfill')
        
        elif handling_type == 'median':
            for col in columns:
                # replace it with the median because if the data is not normally distributed, the mean will be a problem.
                data_frame[col] = data_frame[col].fillna(data_frame[col].median())
        else:
            print("Invalid Value")
    
    def check_outliers(self, columns, data_frame, whis=1.5):
        fig, axes = plt.subplots(3, 3, figsize=(20, 5 * 3))
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            sns.boxplot(data=data_frame, y=col, ax=axes[i], palette='magma', whis=whis)
            axes[i].set_title(f'Boxplot of {col}', fontsize=12)
            axes[i].set_xlabel('')
            axes[i].set_ylabel(col)
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        return fig
    
    def handle_outliers(self, data_frame, column, upper_value=1.5, lower_value=1.5, handle='no'):
        for col in column:
            Q1 = data_frame[col].quantile(0.25)
            Q3 = data_frame[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - (lower_value * IQR)
            upper_bound = Q3 + (upper_value * IQR)
            
            outliers_mask_lower = (data_frame[col] < lower_bound)
            outliers_mask_upper = (data_frame[col] > upper_bound)
            outliers_count_lower = outliers_mask_lower.sum()
            outliers_count_upper = outliers_mask_upper.sum()
            print(f"Number of lower Outliers in {col}: {outliers_count_lower}")
            print(f"Number of upper Outliers in {col}: {outliers_count_upper}")
            
            if handle == 'yes':
                data_frame[col] = np.where(data_frame[col] < lower_bound, lower_bound, data_frame[col])
                data_frame[col] = np.where(data_frame[col] > upper_bound, upper_bound, data_frame[col])
        
        return data_frame
    
    def scaling_data(self, scaler_type, data_frame, features_train, features_test, columns_list):
        if scaler_type == 'standard scaler':
            # Standard Scalar :
            standard_scaler = StandardScaler()
            features_train[columns_list] = standard_scaler.fit_transform(features_train[columns_list])
            features_test[columns_list] = standard_scaler.transform(features_test[columns_list])
        
        elif scaler_type == 'min max scaler':
            # Min Max Scalar :
            min_max_scaler = MinMaxScaler()
            features_train[columns_list] = min_max_scaler.fit_transform(features_train[columns_list])
            features_test[columns_list] = min_max_scaler.transform(features_test[columns_list])
        
        elif scaler_type == 'robust scaler':
            # Robust Scaler :
            robust_scaler = RobustScaler()
            features_train[columns_list] = robust_scaler.fit_transform(features_train[columns_list])
            features_test[columns_list] = robust_scaler.transform(features_test[columns_list])
        else:
            print("There is no scaler type with this name.")
        
        return features_train, features_test
    
    def ordinal_encoding_data(self, features_train, features_test, columns_list):
        # if encoding_type == 'label':
        
        #   label_encoding = LabelEncoder()
        #   features_train=label_encoding.fit_transform(features_train)
        #   features_test=label_encoding.transform(features_test)
        # handling unseen data
        
        all_categories = {}
        for col in columns_list:
            train_cats = features_train[col].unique()
            all_categories[col] = sorted(set(train_cats))
        
        ordinal_encoder = OrdinalEncoder(
            categories=[all_categories[col] for col in columns_list],
            handle_unknown='use_encoded_value',
            unknown_value=len(all_categories[col])
        )
        
        features_train[columns_list] = ordinal_encoder.fit_transform(features_train[columns_list])
        features_test[columns_list] = ordinal_encoder.transform(features_test[columns_list])
        
        return features_train, features_test
    
    def spliting_data(self, data_frame, label, test_size=0.2, random_state=42):
        features = data_frame.drop([label], axis=1)
        target = data_frame[label]
        # random_state = 42 -> to make the split the same every time
        features_train, features_test, target_train, target_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        
        return features_train, features_test, target_train, target_test
    
    def correlation(self, features_train, target_train, data_frame, numerical_columns, categorical_columns):
        # Numerical : anova
        x = features_train[numerical_columns]
        y = target_train
        
        f_values, p_values = f_classif(x, y)
        
        numerical_anova_data_frame = pd.DataFrame({
            'Feature': numerical_columns,
            'F-Score': f_values,
            'P-Value': p_values
        }).sort_values(by='F-Score', ascending=False)
        
        # Categorical : chi2
        x = features_train[categorical_columns]
        y = target_train
        
        chi2_selector = SelectKBest(score_func=chi2, k='all')
        chi2_selector.fit(x, y)
        
        categorical_chi2_data_frame = pd.DataFrame({
            'Feature': x.columns,
            'Chi2 Score': chi2_selector.scores_,
            'P-Value': chi2_selector.pvalues_
        }).sort_values(by='Chi2 Score', ascending=False)
        
        return numerical_anova_data_frame, categorical_chi2_data_frame
    
    def best_parameters(self, text, model_type, params, model, x_train, x_test, y_train, y_test):
        result = {}
        
        # grid Search : to find the best hyperparameters
        grid = GridSearchCV(
            # random_state = 42 -> same initial weights for comparsion
            estimator=model,
            # what parameters to try
            param_grid=params,
            # cross validation
            cv=5,
            # get the accuracy
            scoring='accuracy',
            # run on cpu
            n_jobs=-1,
            
            verbose=1
        )
        
        grid.fit(x_train, y_train)
        
        # best_estimator_ : returns the model with the best hyperparameters found during grid search
        best = grid.best_estimator_
        result[f'Best {text} Model'] = best
        
        y_train_pred = best.predict(x_train)
        y_test_pred = best.predict(x_test)
        
        if model_type == 'classification':
            # calculate metrics -> accuracy , precision , recall , f1 , auc
            metrics_train = {
                # accuracy : (TP + TN) / (TP + TN + FP + FN)
                'accuracy': accuracy_score(y_train, y_train_pred),
                # precision : TP / (TP + FP)
                'precision': precision_score(y_train, y_train_pred),
                # recall : TP / (TP + FN)
                'recall': recall_score(y_train, y_train_pred),
                # f1 score : 2 * (precision * recall) / (precision + recall)
                'f1': f1_score(y_train, y_train_pred),
                # auc : area under the roc curve
                'auc': roc_auc_score(y_train, y_train_pred),
                # best parameters from grid search
                'best_params': grid.best_params_
            }
            metrics_test = {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred),
                'recall': recall_score(y_test, y_test_pred),
                'f1': f1_score(y_test, y_test_pred),
                'auc': roc_auc_score(y_test, y_test_pred),
                'best_params': grid.best_params_
            }
            print(f"Best {text}")
            print(f"Best Parameters: {grid.best_params_}")
            print("Accuracies are based on Accuracy Score:")
            print(f"Train Accuracy: {metrics_train['accuracy']:.4f}")
            print(f"Test Accuracy: {metrics_test['accuracy']:.4f}")
        
        elif model_type == 'regression':
            metrics_train = {
                # R² : proportion of variance explained by the model
                'r2': r2_score(y_train, y_train_pred),
                # MAE : mean absolute error
                'mae': mean_absolute_error(y_train, y_train_pred),
                # MSE : mean squared error
                'mse': mean_squared_error(y_train, y_train_pred),
                # RMSE : root mean squared error
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                # best parameters from grid search
                'best_params': grid.best_params_
            }
            metrics_test = {
                'r2': r2_score(y_test, y_test_pred),
                'mae': mean_absolute_error(y_test, y_test_pred),
                'mse': mean_squared_error(y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'best_params': grid.best_params_
            }
            print(f"Best {text}")
            print(f"Best Parameters: {grid.best_params_}")
            print("Accuracies are based on R² Score:")
            print(f"Train Accuracy: {metrics_train['r2']:.4f}")
            print(f"Test Accuracy: {metrics_test['r2']:.4f}")
        
        elif model_type == 'unsupervised':
            # calculate metrics -> silhouette , davies-bouldin , calinski-harabasz
            metrics_train = {
                # silhouette : cohesion vs separation (-1 to 1, higher is better)
                'silhouette': silhouette_score(x_train, y_train_pred),
                # davies-bouldin : average similarity between clusters (lower is better)
                'davies_bouldin': davies_bouldin_score(x_train, y_train_pred),
                # calinski-harabasz : ratio of between-cluster dispersion to within-cluster dispersion (higher is better)
                'calinski_harabasz': calinski_harabasz_score(x_train, y_train_pred),
                # best parameters from grid search
                'best_params': grid.best_params_
            }
            metrics_test = {
                'silhouette': silhouette_score(x_test, y_test_pred),
                'davies_bouldin': davies_bouldin_score(x_test, y_test_pred),
                'calinski_harabasz': calinski_harabasz_score(x_test, y_test_pred),
                'best_params': grid.best_params_
            }
            print(f"Best {text}")
            print(f"Best Parameters: {grid.best_params_}")
            print("Accuracies are based on Silhouette Score:")
            print(f"Train Accuracy: {metrics_train['silhouette']:.4f}")
            print(f"Test Accuracy: {metrics_test['silhouette']:.4f}")
        
        result[f'{text} Train Metrics'] = metrics_train
        result[f'{text} Test Metrics'] = metrics_test
        
        return result
    
    def train_test_evaluate(self, scaler_type, x_train, x_test, y_train, y_test, model):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred_train = model.predict(x_train)
        print(f"{scaler_type} Scaler")
        print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
        print("Test Accuracy:", accuracy_score(y_test, y_pred))
    
    def evaluate_classification(self, y_test, y_pred, labels=None):
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    
    def evaluate_regression(self, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
    
    def evaluate_clustering(self, X, labels, true_labels=None):
        # Internal metrics (no ground truth needed)
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        
        print(f"Silhouette Score: {sil:.4f}")
        print(f"Davies-Bouldin Index: {db:.4f}")
        
        # External metric (only if true labels are available)
        if true_labels is not None:
            ari = adjusted_rand_score(true_labels, labels)
            print(f"Adjusted Rand Index (vs true labels): {ari:.4f}")
'''

# Create the machine_learning_class.py file
with open("machine_learning_class.py", "w", encoding="utf-8") as f:
    f.write(ml_class_code)

# Main execution
if __name__ == "__main__":
    app = MLDashboard()
    app.run()