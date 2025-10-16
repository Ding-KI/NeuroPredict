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

# 特征名称映射表 - 将技术名称转换为可读名称
FEATURE_NAME_MAPPING = {
    'APQ_P_APQ_P_CP': 'Parenting Corporal Punishment',
    'APQ_P_APQ_P_ID': 'Parenting Inconsistent Discipline', 
    'APQ_P_APQ_P_INV': 'Parenting Involvement',
    'APQ_P_APQ_P_OPD': 'Parenting Other Discipline Practices',
    'APQ_P_APQ_P_PM': 'Parenting Poor Monitoring',
    'APQ_P_APQ_P_PP': 'Parenting Positive Parenting',
    'SDQ_SDQ_Conduct_Problems': 'Conduct Problems',
    'SDQ_SDQ_Difficulties_Total': 'Total Difficulties',
    'SDQ_SDQ_Emotional_Problems': 'Emotional Problems',
    'SDQ_SDQ_Externalizing': 'Externalizing Behavior',
    'SDQ_SDQ_Generating_Impact': 'Impact on Child',
    'SDQ_SDQ_Hyperactivity': 'Hyperactivity',
    'SDQ_SDQ_Internalizing': 'Internalizing Behavior',
    'SDQ_SDQ_Peer_Problems': 'Peer Problems',
    'SDQ_SDQ_Prosocial': 'Prosocial Behavior',
    'Sex_F': 'Gender (Female)'
}

# Create a lock for thread safety
_lock = threading.RLock()

def apply_feature_name_mapping(shap_values, feature_names=None):
    """
    应用特征名称映射，将技术名称转换为可读名称
    """
    if hasattr(shap_values, 'feature_names'):
        # 创建新的feature_names列表
        new_feature_names = []
        for name in shap_values.feature_names:
            new_feature_names.append(FEATURE_NAME_MAPPING.get(name, name))
        
        # 创建新的SHAP Explanation对象
        return shap.Explanation(
            values=shap_values.values,
            base_values=shap_values.base_values,
            data=shap_values.data,
            feature_names=new_feature_names
        )
    return shap_values

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

    sample_index = st.slider( " ", 0, len(X_test) - 1, 5)
    # 创建带有可读列名的DataFrame
    sample_data = X_test.iloc[[sample_index]].copy()
    sample_data.columns = [FEATURE_NAME_MAPPING.get(col, col) for col in sample_data.columns]
    st.dataframe(sample_data, width='stretch')
    
    # Force Plot 
    st.subheader("SHAP Force Plot")
    
    # We use the raw explainer object and the specific sample's SHAP values
    with _lock:
        # 创建更大的图形以容纳更多文本
        plt.figure(figsize=(14, 8))
        
        # 创建SHAP force plot，并设置数值精度和特征名称映射
        shap_values_rounded = shap_values_class_1[sample_index]
        shap_values_rounded.values = np.round(shap_values_rounded.values, 2)
        shap_values_rounded.data = np.round(shap_values_rounded.data, 2)
        
        # 应用特征名称映射
        shap_values_mapped = apply_feature_name_mapping(shap_values_rounded)
        
        # 使用自定义参数来改善文本显示，并将蓝色改为绿色
        shap.plots.force(shap_values_mapped, matplotlib=True, 
                        text_rotation=0, 
                        show=False)
        
        # 获取当前图形并调整布局
        fig = plt.gcf()
        
        # 调整子图参数以提供更多空间给文本
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
        
        # 将蓝色改为绿色
        for ax in fig.get_axes():
            # 修改文本颜色
            for text in ax.texts:
                if text.get_fontsize() > 8:  # 只调整较大的字体
                    text.set_fontsize(9.5)
                # 将蓝色文本改为绿色
                if text.get_color() == '#1f77b4' or 'blue' in str(text.get_color()).lower():
                    text.set_color('#1D5746')  # 使用主题绿色
            
            # 修改图形元素颜色
            for patch in ax.patches:
                if patch.get_facecolor() == '#1f77b4' or 'blue' in str(patch.get_facecolor()).lower():
                    patch.set_facecolor('#1D5746')  # 使用主题绿色
            
            # 修改线条颜色
            for line in ax.lines:
                if line.get_color() == '#1f77b4' or 'blue' in str(line.get_color()).lower():
                    line.set_color('#1D5746')  # 使用主题绿色
        
        st.pyplot(fig)

    # Waterfall Plot
    st.subheader("SHAP Waterfall Plot")
    
    # 使用三列布局将图片固定在中间
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        with _lock:
            plt.figure(figsize=(8, 6))
            # 应用特征名称映射
            shap_values_mapped = apply_feature_name_mapping(shap_values_class_1[sample_index])
            shap.plots.waterfall(shap_values_mapped, max_display=16, show=False)
            fig_waterfall = plt.gcf()
            st.pyplot(fig_waterfall)

with tab2:
    st.header("Model Explanations")

    col1, col2 = st.columns(2)

    with col1:
        # Bar Plot
        st.subheader("Overall Feature Importance")

        with _lock:
            plt.figure(figsize=(10, 6))
            # 应用特征名称映射
            shap_values_mapped = apply_feature_name_mapping(shap_values_class_1)
            shap.plots.bar(shap_values_mapped, max_display=16, show=False)
            fig_bar = plt.gcf()
            st.pyplot(fig_bar)
    with col2:
        # Beeswarm Plot
        st.subheader("SHAP Beeswarm Plot")

        with _lock:
            plt.figure(figsize=(10, 6))
            # 应用特征名称映射
            shap_values_mapped = apply_feature_name_mapping(shap_values_class_1)
            shap.plots.beeswarm(shap_values_mapped, max_display=16, show=False)
            fig_beeswarm = plt.gcf()
            st.pyplot(fig_beeswarm)

    st.markdown( "----")
    st.subheader("In Females")
    col1, col2 = st.columns(2)

    with col1:
         # Bar Plot (Females)
        with _lock:
            plt.figure(figsize=(10, 6))
            # 应用特征名称映射
            shap_values_mapped = apply_feature_name_mapping(explanation_female)
            shap.plots.bar(shap_values_mapped, max_display=16, show=False)
            fig_bar = plt.gcf()
            st.pyplot(fig_bar)

    with col2:
        # Beeswarm Plot (Females)
        with _lock:
            plt.figure(figsize=(10, 6))
            # 应用特征名称映射
            shap_values_mapped = apply_feature_name_mapping(explanation_female)
            shap.plots.beeswarm(shap_values_mapped, max_display=16, show=False)
            fig_beeswarm = plt.gcf()
            st.pyplot(fig_beeswarm)
        
    st.subheader("In Males")
    col1, col2 = st.columns(2)

    with col1:
        # Bar Plot (Males)
        with _lock:
            plt.figure(figsize=(10, 6))
            # 应用特征名称映射
            shap_values_mapped = apply_feature_name_mapping(explanation_male)
            shap.plots.bar(shap_values_mapped, max_display=16, show=False)
            fig_bar = plt.gcf()
            st.pyplot(fig_bar)

    with col2:
        # Beeswarm Plot (Males)
        with _lock:
            plt.figure(figsize=(10, 6))
            # 应用特征名称映射
            shap_values_mapped = apply_feature_name_mapping(explanation_male)
            shap.plots.beeswarm(shap_values_mapped, max_display=16, show=False)
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