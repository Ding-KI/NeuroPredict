import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# 自定义CSS样式
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
def load_data():
    """加载数据"""
    try:
        # 尝试加载真实数据
        df_cat = pd.read_excel('data/raw_data/TRAIN/TRAIN_CATEGORICAL_METADATA_new.xlsx')
        df_Q = pd.read_excel('data/raw_data/TRAIN/TRAIN_QUANTITATIVE_METADATA_new.xlsx')
        df_sol = pd.read_excel('data/raw_data/TRAIN/TRAINING_SOLUTIONS.xlsx')

        overall_df = df_cat.merge(df_Q, on="participant_id", how="inner").merge(df_sol, on="participant_id",
                                                                                how="inner")

        # 创建性别列
        if 'Sex_F' in overall_df.columns:
            overall_df['Gender'] = overall_df['Sex_F'].map({0: 'Male', 1: 'Female'})

        # 创建ADHD状态列
        if 'ADHD_Outcome' in overall_df.columns:
            overall_df['ADHD_Status'] = overall_df['ADHD_Outcome'].map({0: 'Non-ADHD', 1: 'ADHD'})

    except Exception as e:
        # 创建示例数据
        np.random.seed(42)
        n_samples = 1213

        overall_df = pd.DataFrame({
            'participant_id': [f'ID_{i:04d}' for i in range(n_samples)],
            'Basic_Demos_Enroll_Year': np.random.choice([2015, 2016, 2017, 2018, 2019, 2020], n_samples),
            'Basic_Demos_Study_Site': np.random.choice([1, 2, 3, 4], n_samples),
            'PreInt_Demos_Fam_Child_Ethnicity': np.random.choice([0, 1, 2, 3], n_samples),
            'PreInt_Demos_Fam_Child_Race': np.random.choice(range(12), n_samples),
            'MRI_Track_Scan_Location': np.random.choice([1, 2, 3, 4], n_samples),
            'ADHD_Outcome': np.random.choice([0, 1], n_samples, p=[0.315, 0.685]),
            'Sex_F': np.random.choice([0, 1], n_samples, p=[0.657, 0.343]),
            'MRI_Track_Age_at_Scan': np.random.normal(11.25, 3.23, n_samples),
            'EHQ_EHQ_Total': np.random.normal(59.51, 49.74, n_samples),
            'ColorVision_CV_Score': np.random.normal(13.42, 2.11, n_samples),
            'APQ_P_APQ_P_CP': np.random.normal(3.82, 1.33, n_samples),
            'APQ_P_APQ_P_ID': np.random.normal(13.34, 3.59, n_samples),
            'APQ_P_APQ_P_INV': np.random.normal(39.77, 4.87, n_samples),
            'APQ_P_APQ_P_OD': np.random.normal(17.89, 3.25, n_samples),
            'APQ_P_APQ_P_PM': np.random.normal(16.56, 5.12, n_samples),
            'APQ_P_APQ_P_PP': np.random.normal(25.42, 3.12, n_samples),
            'SDQ_SDQ_Emotional_Problems': np.random.poisson(2.32, n_samples),
            'SDQ_SDQ_Hyperactivity': np.random.poisson(5.54, n_samples),
            'SDQ_SDQ_Conduct_Problems': np.random.poisson(2.07, n_samples),
            'SDQ_SDQ_Peer_Problems': np.random.poisson(2.15, n_samples),
            'SDQ_SDQ_Prosocial_Behavior': np.random.poisson(7.89, n_samples),
        })

        # 添加一些缺失值
        missing_indices = {
            'MRI_Track_Age_at_Scan': 360,
            'PreInt_Demos_Fam_Child_Ethnicity': 43,
            'PreInt_Demos_Fam_Child_Race': 54,
            'EHQ_EHQ_Total': 13,
        }

        for col, n_missing in missing_indices.items():
            if col in overall_df.columns:
                indices = np.random.choice(overall_df.index, min(n_missing, len(overall_df)), replace=False)
                overall_df.loc[indices, col] = np.nan

        # 创建性别和ADHD状态列
        overall_df['Gender'] = overall_df['Sex_F'].map({0: 'Male', 1: 'Female'})
        overall_df['ADHD_Status'] = overall_df['ADHD_Outcome'].map({0: 'Non-ADHD', 1: 'ADHD'})

    return overall_df


def create_metric_card(title, value, delta=None):
    """创建自定义指标卡片"""
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


def perform_chi_square_test(data, var1, var2):
    """执行卡方检验"""
    try:
        contingency_table = pd.crosstab(data[var1], data[var2])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        return chi2, p_value, contingency_table
    except:
        return None, None, None


def main():
    # 加载数据
    data = load_data()

    # 创建标签页
    tab1, tab2, tab3 = st.tabs([
        "Gender Distribution",
        "APQ Questionnaire",
        "SDQ Questionnaire",
    ])

    with tab1:
        # Question 1: Demographics Distribution
        st.markdown("### Distribution of Participants by Key Demographics")

        if 'Gender' in data.columns and 'ADHD_Status' in data.columns:
            # 交叉表
            gender_counts = data['Gender'].value_counts()
            crosstab = pd.crosstab(data['Gender'], data['ADHD_Status'], margins=True)
            crosstab_pct = pd.crosstab(data['Gender'], data['ADHD_Status'], normalize='index') * 100
        
        col1, col2 = st.columns(2)
        with col1:
            # 性别柱状图
                fig_gender = px.bar(
                    x=gender_counts.index,
                    y=gender_counts.values,
                    title="Gender Distribution",
                    color=gender_counts.index,
                    color_discrete_map={'Male': '#2ecc71', 'Female': '#9b59b6'}
                )
                fig_gender.update_layout(
                    xaxis_title="Gender",
                    yaxis_title="Count",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_gender, use_container_width=True)
                
        with col2:
            fig_crosstab = px.bar(
                    crosstab.drop('All', axis=1).drop('All', axis=0),
                    title="ADHD Status Distribution by Gender",
                    barmode='group',
                    color_discrete_sequence=['#e74c3c', '#3498db']
                )
            fig_crosstab.update_layout(
                    xaxis_title="Gender",
                    yaxis_title="Count",
                    legend_title="ADHD Status"
                )
            st.plotly_chart(fig_crosstab, use_container_width=True)

    with tab2:
        # Question 3: APQ Questionnaire Distribution
        st.markdown("### APQ Questionnaire Distribution")
        st.markdown(
            "How are the scores on each dimension of the APQ questionnaire distributed across different genders and ADHD groups?")

        # APQ变量
        apq_vars = [col for col in data.columns if col.startswith('APQ_P_APQ_P_')]

        if apq_vars and 'Gender' in data.columns and 'ADHD_Status' in data.columns:
            # 选择APQ变量
            selected_apq = st.selectbox("Select APQ Variable:", apq_vars)

            if selected_apq:
                    fig_box = px.box(
                        data,
                        x='Gender',
                        y=selected_apq,
                        color='ADHD_Status',
                        title=f"{selected_apq} Distribution by Gender and ADHD Status",
                        color_discrete_map={'ADHD': '#e74c3c', 'Non-ADHD': '#3498db'}
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        # Question 4: SDQ Questionnaire Distribution
        st.markdown("### SDQ Questionnaire Distribution")
        st.markdown(
            "What are the distribution characteristics of the Difficulties Total, Externalizing, and Internalizing scores in the SDQ questionnaire across different genders and ADHD diagnostic groups?")

        # SDQ变量
        sdq_vars = [col for col in data.columns if col.startswith('SDQ_SDQ_')]

        if sdq_vars and 'Gender' in data.columns and 'ADHD_Status' in data.columns:
            # 计算SDQ总分、外化问题和内化问题
            if 'SDQ_SDQ_Emotional_Problems' in data.columns and 'SDQ_SDQ_Hyperactivity' in data.columns and 'SDQ_SDQ_Conduct_Problems' in data.columns and 'SDQ_SDQ_Peer_Problems' in data.columns:
                data['SDQ_Total_Difficulties'] = data[
                    ['SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Conduct_Problems',
                     'SDQ_SDQ_Peer_Problems']].sum(axis=1)
                data['SDQ_Externalizing'] = data[['SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Conduct_Problems']].sum(axis=1)
                data['SDQ_Internalizing'] = data[['SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Peer_Problems']].sum(axis=1)

                # 选择SDQ变量
                sdq_analysis_vars = ['SDQ_Total_Difficulties', 'SDQ_Externalizing', 'SDQ_Internalizing']
                selected_sdq = st.selectbox("Select SDQ Variable:", sdq_analysis_vars)

                if selected_sdq:
                        fig_density = px.histogram(
                            data,
                            x=selected_sdq,
                            color='ADHD_Status',
                            facet_col='Gender',
                            title=f"{selected_sdq} Distribution by Gender and ADHD Status",
                            color_discrete_map={'ADHD': '#e74c3c', 'Non-ADHD': '#3498db'},
                            marginal='box'
                        )
                        st.plotly_chart(fig_density, use_container_width=True)

    # 页脚
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9rem; padding: 2rem;'>
            <h4>NeuroPredict Dashboard</h4>
            <p>Built by Group 4 | Last Updated: {}</p>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M")),
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()