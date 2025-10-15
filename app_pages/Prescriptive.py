import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import shap
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
import threading

# Create a lock for thread safety
_lock = threading.RLock()

st.markdown("""
<style>
/* 保持默认的绿色边框 */
.stTabs [data-baseweb="tab-border"] {
    background-color: #1D5746 !important;
}

/* 选中的tab指示条改为红色 */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #e74c3c !important;
}

/* 选中的tab文字改为红色 */
.stTabs [aria-selected="true"] {
    color: #e74c3c !important;
}

/* 未选中的tab文字保持绿色 */
.stTabs [data-baseweb="tab"] {
    color: #1D5746 !important;
}

/* 悬浮时变成红色 */
.stTabs [data-baseweb="tab"]:hover {
    color: #e74c3c !important;
}

.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.section-header {
    font-size: 1.8rem;
    color: #2c3e50;
    margin-top: 2rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid #e74c3c;
    padding-bottom: 0.5rem;
}

.question-header {
    font-size: 1.4rem;
    color: #34495e;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}

.metric-card {
    background: #1D5746;
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 15px rgba(29, 87, 70, 0.4);
    margin: 0.5rem;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

.metric-title {
    font-size: 0.9rem;
    opacity: 0.9;
    margin-bottom: 0.5rem;
    color: white;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: white;
}

.info-box {
    background: #1D5746;
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.statistics-box {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #007bff;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def compute_shap_values():
    """
    Loads model and data, then computes the SHAP explainer and values.
    """
    try:
        model = joblib.load('model/best_model_Decision_Tree_Depth=3.joblib')
        df = pd.read_csv('data/processed_data/df_preprocessed.csv')

        ADHD_Outcome = 'ADHD_Outcome' 
        if ADHD_Outcome not in df.columns:
            st.error(f"Target column '{ADHD_Outcome}' not found in the dataset.")
            return None, None, None, None

        X = df.drop(columns=[ADHD_Outcome])
        y = df[ADHD_Outcome]
        _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Create the SHAP explainer
        explainer = shap.TreeExplainer(model)
        # Compute SHAP values for the test set
        shap_values = explainer(X_test) 
        return model, X_test, explainer, shap_values

    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Make sure 'model.joblib' and 'dataset.csv' are in the same directory as 'app.py'.")
        return None, None, None, None
    except Exception as e:
        st.error(f"An error occurred during computation: {e}")
        return None, None, None, None

with st.spinner('Loading model and computing SHAP values... This may take a moment.'):
    model, X_test, explainer, shap_values = compute_shap_values()

if model is None:
    st.stop()

shap_values_class_1 = shap_values[:,:,1]
# Identify indices for female and male samples
female_indices = X_test['Sex_F'] == 1
male_indices = X_test['Sex_F'] == 0
explanation_female = shap_values_class_1[female_indices.values]
explanation_male = shap_values_class_1[male_indices.values]

tab1, tab2 = st.tabs(["Single Prediction Explanation", "Global Model Explanations"])

with tab1:
    st.header("Explain a Single Prediction")
    st.markdown("Select a sample from the test set using the slider to see how the model made its prediction.")

    sample_index = st.slider("Select a sample index:", 0, len(X_test) - 1, 5)

    st.subheader(f"Features for Sample #{sample_index}")
    st.dataframe(X_test.iloc[[sample_index]], use_container_width=True)
    
    # Force Plot 
    st.subheader("SHAP Force Plot")
    st.markdown("This plot shows which feature values pushed the prediction up (in red) or down (in blue).")
    
    # We use the raw explainer object and the specific sample's SHAP values
    with _lock:
        plt.figure(figsize=(10, 6))
        shap.plots.force(shap_values_class_1[sample_index], matplotlib=True)
        fig = plt.gcf()
        st.pyplot(fig)

    # Waterfall Plot
    st.subheader("SHAP Waterfall Plot")
    st.markdown("This plot breaks down the contribution of each feature to the final prediction.")
    
    with _lock:
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(shap_values_class_1[sample_index], max_display=16, show=False)
        fig_waterfall = plt.gcf()
        st.pyplot(fig_waterfall)

with tab2:
    st.header("Model Explanations")
    st.markdown("These plots summarize the model's behavior across the entire test set.")

    col1, col2 = st.columns(2)

    with col1:
        # Bar Plot
        st.subheader("Overall Feature Importance")
        st.markdown("This plot shows the average impact of each feature on the model's predictions.")

        with _lock:
            plt.figure(figsize=(10, 6))
            shap.plots.bar(shap_values_class_1, max_display=16, show=False)
            fig_bar = plt.gcf()
            st.pyplot(fig_bar)
    with col2:
        # Beeswarm Plot
        st.subheader("SHAP Beeswarm Plot")
        st.markdown("This plot shows the SHAP value for every feature for every sample. The color indicates the feature's value.")

        with _lock:
            plt.figure(figsize=(10, 6))
            shap.plots.beeswarm(shap_values_class_1, max_display=16, show=False)
            fig_beeswarm = plt.gcf()
            st.pyplot(fig_beeswarm)

    st.markdown( "----")
    st.subheader("In Females")
    col1, col2 = st.columns(2)

    with col1:
         # Bar Plot (Females)
        with _lock:
            plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_female, max_display=16, show=False)
            fig_bar = plt.gcf()
            st.pyplot(fig_bar)

    with col2:
        # Beeswarm Plot (Females)
        with _lock:
            plt.figure(figsize=(10, 6))
            shap.plots.beeswarm(explanation_female, max_display=16, show=False)
            fig_beeswarm = plt.gcf()
            st.pyplot(fig_beeswarm)
        
    st.subheader("In Males")
    col1, col2 = st.columns(2)

    with col1:
        # Bar Plot (Males)
        with _lock:
            plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_male, max_display=16, show=False)
            fig_bar = plt.gcf()
            st.pyplot(fig_bar)

    with col2:
        # Beeswarm Plot (Males)
        with _lock:
            plt.figure(figsize=(10, 6))
            shap.plots.beeswarm(explanation_male, max_display=16, show=False)
            fig_beeswarm = plt.gcf()
            st.pyplot(fig_beeswarm)

st.markdown("----")
st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9rem; padding: 2rem;'>
            <h4>NeuroPredict Dashboard</h4>
            <p>Built by Group 4 | Last Updated: {}</p>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M")),
        unsafe_allow_html=True
    )