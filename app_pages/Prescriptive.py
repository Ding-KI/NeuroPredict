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

_lock = threading.RLock()

def apply_feature_name_mapping(shap_values, feature_names=None):
    if hasattr(shap_values, 'feature_names'):
        new_feature_names = []
        for name in shap_values.feature_names:
            new_feature_names.append(FEATURE_NAME_MAPPING.get(name, name))
        
        return shap.Explanation(
            values=shap_values.values,
            base_values=shap_values.base_values,
            data=shap_values.data,
            feature_names=new_feature_names
        )
    return shap_values

st.markdown("""
<style>
.stTabs [data-baseweb="tab-border"] {
    background-color: #1D5746 !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: #e74c3c !important;
}

.stTabs [aria-selected="true"] {
    color: #e74c3c !important;
}

.stTabs [data-baseweb="tab"] {
    color: #1D5746 !important;
}

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
    X, y, model, df, X_test, explainer, shap_values = None, None, None, None, None, None, None
    
    try:
        model = joblib.load('model/best_model_Decision_Tree_Depth=3.joblib')
        df = pd.read_csv('data/processed_data/df_preprocessed.csv')

        ADHD_Outcome = 'ADHD_Outcome' 
        if ADHD_Outcome not in df.columns:
            st.error(f"Target column '{ADHD_Outcome}' not found in the dataset.")
            return None, None, None, None

        X = df.drop(columns=[ADHD_Outcome])
        y = df[ADHD_Outcome]

        try:
            X = X.astype(float)
        except ValueError as e:
            st.error(f"Error converting feature data to numeric: {e}. Please check 'df_preprocessed.csv' for non-numeric values (like text) in your feature columns.")
            return None, None, None, None
        
        _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
        
        #st.write(f"Debug: SHAP values shape: {shap_values.values.shape}")
        
        return model, X_test, explainer, shap_values

    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Make sure 'model/best_model_Decision_Tree_Depth=3.joblib' and 'data/processed_data/df_preprocessed.csv' exist.")
        return None, None, None, None
    except Exception as e:
        st.error(f"An error occurred during computation: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None

with st.spinner('Loading model and computing SHAP values... This may take a moment.'):
    model, X_test, explainer, shap_values = compute_shap_values()

if model is None:
    st.stop()

try:
    if len(shap_values.values.shape) == 3:
        shap_values_class_1 = shap.Explanation(
            values=shap_values.values[:, :, 1],
            base_values=shap_values.base_values[:, 1] if len(shap_values.base_values.shape) > 1 else shap_values.base_values,
            data=shap_values.data,
            feature_names=shap_values.feature_names
        )
    else:
        shap_values_class_1 = shap_values
except Exception as e:
    st.error(f"Error processing SHAP values: {e}")
    st.stop()

try:
    female_mask = (X_test['Sex_F'].values == 1)
    male_mask = (X_test['Sex_F'].values == 0)
    
    explanation_female = shap.Explanation(
        values=shap_values_class_1.values[female_mask],
        base_values=shap_values_class_1.base_values[female_mask] if hasattr(shap_values_class_1.base_values, '__len__') else shap_values_class_1.base_values,
        data=shap_values_class_1.data[female_mask],
        feature_names=shap_values_class_1.feature_names
    )
    
    explanation_male = shap.Explanation(
        values=shap_values_class_1.values[male_mask],
        base_values=shap_values_class_1.base_values[male_mask] if hasattr(shap_values_class_1.base_values, '__len__') else shap_values_class_1.base_values,
        data=shap_values_class_1.data[male_mask],
        feature_names=shap_values_class_1.feature_names
    )
except Exception as e:
    st.error(f"Error creating gender-specific explanations: {e}")
    import traceback
    st.error(traceback.format_exc())
    st.stop()

tab1, tab2 = st.tabs(["Single Prediction Explanation", "Global Model Explanations"])

with tab1:
    st.header("Explain a Single Prediction")
    st.markdown("Select a sample from the test set using the slider to see how the model made its prediction.")

    sample_index = st.slider(" ", 0, len(X_test) - 1, 5)
    sample_data = X_test.iloc[[sample_index]].copy()
    sample_data.columns = [FEATURE_NAME_MAPPING.get(col, col) for col in sample_data.columns]
    st.dataframe(sample_data, use_container_width=True)
    
    # Force Plot 
    st.subheader("SHAP Force Plot")
    
    with _lock:
        try:
            plt.figure(figsize=(14, 8))
            
            sample_shap = shap.Explanation(
                values=shap_values_class_1.values[sample_index],
                base_values=shap_values_class_1.base_values[sample_index] if hasattr(shap_values_class_1.base_values, '__len__') else shap_values_class_1.base_values,
                data=shap_values_class_1.data[sample_index],
                feature_names=shap_values_class_1.feature_names
            )
            
            sample_shap.values = np.round(sample_shap.values, 2)
            sample_shap.data = np.round(sample_shap.data, 2)
            
            shap_values_mapped = apply_feature_name_mapping(sample_shap)
            
            shap.plots.force(shap_values_mapped, matplotlib=True, 
                            text_rotation=0, 
                            show=False)
            
            fig = plt.gcf()
            
            plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
            
            for ax in fig.get_axes():
                for text in ax.texts:
                    if text.get_fontsize() > 8:  # 只调整较大的字体
                        text.set_fontsize(9.5)
                    if text.get_color() == '#1f77b4' or 'blue' in str(text.get_color()).lower():
                        text.set_color('#1D5746')  # 使用主题绿色
                
                for patch in ax.patches:
                    if patch.get_facecolor() == '#1f77b4' or 'blue' in str(patch.get_facecolor()).lower():
                        patch.set_facecolor('#1D5746')  # 使用主题绿色
                
                for line in ax.lines:
                    if line.get_color() == '#1f77b4' or 'blue' in str(line.get_color()).lower():
                        line.set_color('#1D5746')  # 使用主题绿色
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating force plot: {e}")
            import traceback
            st.error(traceback.format_exc())

    # Waterfall Plot
    st.subheader("SHAP Waterfall Plot")
    
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        with _lock:
            try:
                plt.figure(figsize=(8, 6))
                
                sample_shap = shap.Explanation(
                    values=shap_values_class_1.values[sample_index],
                    base_values=shap_values_class_1.base_values[sample_index] if hasattr(shap_values_class_1.base_values, '__len__') else shap_values_class_1.base_values,
                    data=shap_values_class_1.data[sample_index],
                    feature_names=shap_values_class_1.feature_names
                )
                
                shap_values_mapped = apply_feature_name_mapping(sample_shap)
                shap.plots.waterfall(shap_values_mapped, max_display=16, show=False)
                fig_waterfall = plt.gcf()
                st.pyplot(fig_waterfall)
            except Exception as e:
                st.error(f"Error creating waterfall plot: {e}")
                import traceback
                st.error(traceback.format_exc())

with tab2:
    st.header("Model Explanations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Overall Feature Importance")

        with _lock:
            try:
                plt.figure(figsize=(10, 6))
                # 应用特征名称映射
                shap_values_mapped = apply_feature_name_mapping(shap_values_class_1)
                shap.plots.bar(shap_values_mapped, max_display=16, show=False)
                fig_bar = plt.gcf()
                st.pyplot(fig_bar)
            except Exception as e:
                st.error(f"Error creating bar plot: {e}")
                
    with col2:
        # Beeswarm Plot
        st.subheader("SHAP Beeswarm Plot")

        with _lock:
            try:
                plt.figure(figsize=(10, 6))
                # 应用特征名称映射
                shap_values_mapped = apply_feature_name_mapping(shap_values_class_1)
                shap.plots.beeswarm(shap_values_mapped, max_display=16, show=False)
                fig_beeswarm = plt.gcf()
                st.pyplot(fig_beeswarm)
            except Exception as e:
                st.error(f"Error creating beeswarm plot: {e}")

    st.markdown("----")
    st.subheader("In Females")
    col1, col2 = st.columns(2)

    with col1:
         # Bar Plot (Females)
        with _lock:
            try:
                plt.figure(figsize=(10, 6))
                # 应用特征名称映射
                shap_values_mapped = apply_feature_name_mapping(explanation_female)
                shap.plots.bar(shap_values_mapped, max_display=16, show=False)
                fig_bar = plt.gcf()
                st.pyplot(fig_bar)
            except Exception as e:
                st.error(f"Error creating female bar plot: {e}")

    with col2:
        # Beeswarm Plot (Females)
        with _lock:
            try:
                plt.figure(figsize=(10, 6))
                # 应用特征名称映射
                shap_values_mapped = apply_feature_name_mapping(explanation_female)
                shap.plots.beeswarm(shap_values_mapped, max_display=16, show=False)
                fig_beeswarm = plt.gcf()
                st.pyplot(fig_beeswarm)
            except Exception as e:
                st.error(f"Error creating female beeswarm plot: {e}")
        
    st.subheader("In Males")
    col1, col2 = st.columns(2)

    with col1:
        # Bar Plot (Males)
        with _lock:
            try:
                plt.figure(figsize=(10, 6))
                # 应用特征名称映射
                shap_values_mapped = apply_feature_name_mapping(explanation_male)
                shap.plots.bar(shap_values_mapped, max_display=16, show=False)
                fig_bar = plt.gcf()
                st.pyplot(fig_bar)
            except Exception as e:
                st.error(f"Error creating male bar plot: {e}")

    with col2:
        # Beeswarm Plot (Males)
        with _lock:
            try:
                plt.figure(figsize=(10, 6))
                # 应用特征名称映射
                shap_values_mapped = apply_feature_name_mapping(explanation_male)
                shap.plots.beeswarm(shap_values_mapped, max_display=16, show=False)
                fig_beeswarm = plt.gcf()
                st.pyplot(fig_beeswarm)
            except Exception as e:
                st.error(f"Error creating male beeswarm plot: {e}")

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