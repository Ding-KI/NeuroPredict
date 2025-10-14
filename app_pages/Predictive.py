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
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

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

@st.cache_data
def load_model(model_path):
    """Loads the pre-trained model"""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model('/Users/kaviya/Repositories/ProHi/NeuroPredict copy/model/best_model_Decision_Tree_Depth=3.joblib')

st.title("Predicting outcome of ADHD diagnosis")
st.markdown("""This page consists of the Alabama Parenting Questionnaire (APQ) and the Strengths and Difficulties Questionnaire (SDQ) to predict the likelihood of ADHD diagnosis in children.
            Please answer all the questions below to the best of your ability.""")
with st.form("questionnaire_form"):
    st.header("Alabama Parenting Questionnaire (APQ)")
    apq_questions = [
    "I praise my child when they behave well.",
    "I find it hard to discipline my child consistently.",
    "I supervise my child’s activities.",
    "I lose my temper when disciplining my child.",]
    apq_responses = []
    for q in apq_questions:
        response = st.slider(q, min_value=1, max_value=5, value=3)
        apq_responses.append(response)
    
    # Question 2 
    st.header("Strengths and Difficulties Questionnaire (SDQ)")
    sdq_questions = [
    "Considerate of other people’s feelings.",
    "Restless, overactive, cannot stay still for long.",
    "Often complains of headaches, stomach aches or sickness.",
    "Shares readily with other children.",]
    
    sdq_responses = []
    for q in sdq_questions:
        response = st.slider(q, min_value=0, max_value=2, value=1)
        sdq_responses.append(response)

    submitted = st.form_submit_button("Predict Outcome")
features = np.array(apq_responses + sdq_responses).reshape(1, -1)
st.divider()

if submitted:
        prediction = model.predict(features)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0][1]
        
        st.subheader("Prediction Result:")
        
        if prediction[0] == 1:
            st.error("High likelihood of ADHD symptoms detected.")
        else:
            st.success("Low likelihood of ADHD symptoms detected.")
            
        if proba is not None:
            st.write(f"Model confidence: {proba:.2f}")
    
st.markdown("---")
st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9rem; padding: 2rem;'>
            <h4>NeuroPredict Dashboard</h4>
            <p>Built by Group 4 | Last Updated: {}</p>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M")),
        unsafe_allow_html=True
    )
