import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import chi2_contingency
import warnings

warnings.filterwarnings("ignore")

# 自定义CSS样式
st.markdown("""
<style>
<style>

    /* Tab 选中指示条颜色 */
    .stTabs [data-baseweb="tab-highlight"] {
    background-color: #1D5746 !important;}
    
    .stTabs [data-baseweb="tab-border"] {
    background-color: #1D5746 !important;}
    
    /* 将选中标签的红色文字改为绿色 */
    .stTabs [aria-selected="true"] {
    color: #1D5746 !important;}
    
    /* 将所有标签文字改为绿色 */
    .stTabs [data-baseweb="tab"] {
    color: #1D5746 !important;}

    /* 确保选中状态也是绿色 */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #1D5746 !important;}

    /* 悬浮时变成红色 */
    .stTabs [data-baseweb="tab"]:hover {
    color: #e74c3c !important;}
    

    
    .main-header {
        font-size: 3rem;
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

    .metric-card {
        background: #1D5746;  /* 修改这里的背景色 */
        padding: 1.5rem;
        border-radius: 15px;
        color: white;  /* 文字改为白色以便于阅读 */
        text-align: center;
        box-shadow: 0 4px 15px rgba(29, 87, 70, 0.4);  /* 调整阴影颜色 */
        margin: 0.5rem;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }

    .metric-title {
        font-size: 0.9rem;
        opacity: 0.9;  /* 稍微调整透明度 */
        margin-bottom: 0.5rem;
        color: white;  /* 确保标题也是白色 */
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: white;  /* 确保数值也是白色 */
    }

    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }

    .navigation-card {
        background: #1D5746;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        cursor: pointer;
        transition: all 0.3s ease;
        color: white;
    }

    .navigation-card:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 25px rgba(0,0,0,0.2);
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    .stApp > header {
        background-color: transparent;
    }

    .stApp {
        margin-top: -80px;
    }

    hr {
        display: none !important;
    }

    .css-1dp5vir {
        background-image: none;
    }
</style>
""", unsafe_allow_html=True)


# 数据加载函数
@st.cache_data
def load_dashboard_data():
    """加载仪表板数据"""
    try:
        # 尝试加载真实数据
        df_cat = pd.read_excel('data/raw_data/TRAIN/TRAIN_CATEGORICAL_METADATA_new.xlsx')
        df_Q = pd.read_excel('data/raw_data/TRAIN/TRAIN_QUANTITATIVE_METADATA_new.xlsx')
        df_sol = pd.read_excel('data/raw_data/TRAIN/TRAINING_SOLUTIONS.xlsx')

        overall_df = df_cat.merge(df_Q, on="participant_id", how="inner").merge(df_sol, on="participant_id",
                                                                                how="inner")

        # 创建性别和ADHD状态列
        if 'Sex_F' in overall_df.columns:
            overall_df['Gender'] = overall_df['Sex_F'].map({0: 'Male', 1: 'Female'})

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
            'Barratt_Barratt_P1_Edu': np.random.normal(17.86, 3.51, n_samples),
            'Barratt_Barratt_P1_Occ': np.random.normal(25.55, 16.76, n_samples),
            'Barratt_Barratt_P2_Edu': np.random.normal(16.88, 3.93, n_samples),
            'Barratt_Barratt_P2_Occ': np.random.normal(30.26, 13.90, n_samples),
        })

        # 添加一些缺失值
        missing_indices = {
            'MRI_Track_Age_at_Scan': 360,
            'PreInt_Demos_Fam_Child_Ethnicity': 43,
            'PreInt_Demos_Fam_Child_Race': 54,
            'EHQ_EHQ_Total': 13,
            'Barratt_Barratt_P2_Edu': 198,
            'Barratt_Barratt_P2_Occ': 222,
        }

        for col, n_missing in missing_indices.items():
            if col in overall_df.columns:
                indices = np.random.choice(overall_df.index, min(n_missing, len(overall_df)), replace=False)
                overall_df.loc[indices, col] = np.nan

        # 创建性别和ADHD状态列
        overall_df['Gender'] = overall_df['Sex_F'].map({0: 'Male', 1: 'Female'})
        overall_df['ADHD_Status'] = overall_df['ADHD_Outcome'].map({0: 'Non-ADHD', 1: 'ADHD'})

    return overall_df


def create_metric_card(title, value, delta=None, delta_color="normal"):
    """创建自定义指标卡片"""
    delta_html = ""
    if delta is not None:
        color = "green" if delta_color == "normal" else "red"
        delta_html = f'<div style="color: {color}; font-size: 0.8rem;">{"↑" if delta > 0 else "↓"} {abs(delta)}</div>'

    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def analyze_missing_values(data):
    """分析缺失值"""
    missing_data = data.isnull().sum()
    missing_percentage = (missing_data / len(data) * 100).round(2)

    missing_df = pd.DataFrame({
        'Feature': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percentage': missing_percentage.values
    }).sort_values('Missing_Percentage', ascending=False)

    return missing_df[missing_df['Missing_Count'] > 0]


def detect_outliers_iqr(data, column):
    """使用IQR方法检测异常值"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


def calculate_correlation_matrix(data, variables):
    """计算相关性矩阵"""
    numeric_data = data[variables].select_dtypes(include=[np.number])
    return numeric_data.corr()


def perform_chi_square_test(data, var1, var2):
    """执行卡方检验"""
    try:
        contingency_table = pd.crosstab(data[var1], data[var2])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        return chi2, p_value, contingency_table
    except:
        return None, None, None


def main():
    # 加载数据并在侧边栏显示统计信息
    data = load_dashboard_data()

    with st.sidebar:

        st.markdown("""
        <h3 style="
            font-size: 1.8rem; 
            color: #FFFFFF; 
            font-weight: 600; 
            margin-bottom: 0rem; 
            margin-top: 0.5rem;
        ">Quick Stats</h3>
        """, unsafe_allow_html=True)
        adhd_rate = (data['ADHD_Outcome'].sum() / len(data) * 100) if 'ADHD_Outcome' in data.columns else 0
        female_rate = (data['Sex_F'].sum() / len(data) * 100) if 'Sex_F' in data.columns else 0

        # 左对齐并增加左边距
        with st.container():
            st.markdown(f"""
            <div style="background-color: #1D5746; padding: 0rem 0rem 0rem 0rem; border-radius: 10px; margin: 0rem 0; text-align: left; box-shadow: 0 2px 10px rgba(29, 87, 70, 0.3);">
                <p style="color: white; margin: 0; font-size: 1.2rem; opacity: 0.9;">ADHD Rate</p>
                <h2 style="color: white; margin: 0; font-size: 2rem;">{adhd_rate:.1f}%</h2>

            </div>
            """, unsafe_allow_html=True)

        with st.container():
            st.markdown(f"""
            <div style="background-color: #1D5746; padding: 0rem 0rem 0rem 0rem; border-radius: 10px; margin: 0rem 0; text-align: left; box-shadow: 0 2px 10px rgba(29, 87, 70, 0.3);">
                <p style="color: white; margin: 0; font-size: 1.2rem; opacity: 0.9;">Female Participants</p>
                <h2 style="color: white; margin: 0; font-size: 2rem;">{female_rate:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
    # 主要内容区域
    tab1, tab2, tab3 = st.tabs(["Overview", "Analytics", "Navigation"])

    with tab1:
        # EDA概览标签
        st.subheader("About NeuroPredict")
        st.markdown("This project aims to use machine learning techniques to predict ADHD diagnosis based on a comprehensive dataset of demographic, behavioral, and clinical features.")

        if 'ADHD_Outcome' in data.columns:
            st.subheader("Target Variables Distribution")
            adhd_counts = data['ADHD_Outcome'].value_counts()
            total_samples = len(data)
            non_adhd_count = adhd_counts.get(0, 0)
            adhd_count = adhd_counts.get(1, 0)
            non_adhd_pct = (non_adhd_count / total_samples) * 100
            adhd_pct = (adhd_count / total_samples) * 100

        col1, col2 = st.columns(2)

        with col1:
            if 'ADHD_Outcome' in data.columns:
                adhd_counts = data['ADHD_Outcome'].value_counts()
                fig_adhd = px.bar(
                    x=['Non-ADHD', 'ADHD'],
                    y=adhd_counts.values.tolist(),
                    title="Distribution of Type of Diagnosis",
                    color=['Non-ADHD', 'ADHD'],
                    color_discrete_map={'Non-ADHD': '#95a5a6', 'ADHD': '#f1c40f'},
                    labels={'x': 'Diagnosis Type', 'y': 'Count'}
                )
                st.plotly_chart(fig_adhd, use_container_width=True)

        with col2:
            if 'Sex_F' in data.columns:
                sex_counts = data['Sex_F'].value_counts()
                fig_sex = px.bar(
                    x=['Male', 'Female'],
                    y=[sex_counts.get(0, 0), sex_counts.get(1, 0)],
                    title="Distribution of Sex of participant",
                    color=['Male', 'Female'],
                    color_discrete_map={'Male': '#74b9ff', 'Female': '#fd79a8'},
                    labels={'x': 'Gender', 'y': 'Count'}
                )
                st.plotly_chart(fig_sex, use_container_width=True)

    with tab2:
      
        if 'MRI_Track_Age_at_Scan' in data.columns:
            st.subheader("Age Distribution Analysis")

            col1, col2 = st.columns(2)

            with col1:
                age_data = data['MRI_Track_Age_at_Scan'].dropna()
                fig_age_hist = px.histogram(
                    x=age_data,
                    nbins=30,
                    title="Age Distribution",
                    labels={'x': 'Age at Scan', 'y': 'Frequency'}
                )
                st.plotly_chart(fig_age_hist, use_container_width=True)

            with col2:
                if 'ADHD_Outcome' in data.columns:
                    fig_age_box = px.box(
                        data,
                        x='ADHD_Outcome',
                        y='MRI_Track_Age_at_Scan',
                        title="Age Distribution by ADHD Outcome",
                        labels={'ADHD_Outcome': 'ADHD Status', 'MRI_Track_Age_at_Scan': 'Age at Scan'}
                    )
                    fig_age_box.update_layout(
                        xaxis=dict(
                            tickvals=[0, 1],
                            ticktext=['No ADHD', 'ADHD']
                        )
                    )
                    st.plotly_chart(fig_age_box, use_container_width=True)

        # SDQ分数分析
        sdq_columns = [col for col in data.columns if col.startswith('SDQ_')]
        if sdq_columns:
            st.subheader("SDQ Scores Analysis")

            sdq_data = data[sdq_columns].mean()

            fig_sdq = px.bar(
                x=sdq_data.index.tolist(),
                y=sdq_data.values.tolist(),
                title="Average SDQ Scores",
                labels={'x': 'SDQ Measures', 'y': 'Average Score'}
            )
            fig_sdq.update_layout(xaxis=dict(tickangle=45))
            st.plotly_chart(fig_sdq, use_container_width=True)

    with tab3:
        # 导航卡片
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="navigation-card">
                <h3>Descriptive Analysis</h3>
                <ul style="text-align: left; margin-top: 1rem;">
                    <li>Dataset Overview</li>
                    <li>Missing Values Analysis</li>
                    <li>Numerical & Categorical Features</li>
                    <li>Target Variables Distribution</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="navigation-card">
                <h3>Diagnostic Analysis</h3>
                <ul style="text-align: left; margin-top: 1rem;">
                    <li>Data Quality Assessment</li>
                    <li>Outlier Detection</li>
                    <li>Distribution Analysis</li>
                    <li>Statistical Tests</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="navigation-card">
                <h3>Predictive Analysis</h3>
                <ul style="text-align: left; margin-top: 1rem;">
                    <li>Feature Engineering</li>
                    <li>Model Training</li>
                    <li>Performance Evaluation</li>
                    <li>Feature Importance</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="navigation-card">
                <h3>Prescriptive Analysis</h3>
                <ul style="text-align: left; margin-top: 1rem;">
                    <li>Treatment Recommendations</li>
                    <li>Risk Stratification</li>
                    <li>Intervention Planning</li>
                    <li>Clinical Decision Support</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # 页脚
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

if __name__ == "__main__":
    main()