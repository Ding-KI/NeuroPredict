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
    tab1, tab2, tab3, tab4 = st.tabs(["EDA Overview", "Analytics", "Data Quality", "Navigation"])

    with tab1:
        # EDA概览标签
        st.subheader("Exploratory Data Analysis Overview")

        # 关键指标
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(create_metric_card("Total Participants", f"{len(data):,}"), unsafe_allow_html=True)

        with col2:
            st.markdown(create_metric_card("Features", f"{len(data.columns):,}"), unsafe_allow_html=True)

        with col3:
            missing_rate = (data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100)
            st.markdown(create_metric_card("Missing Rate", f"{missing_rate:.1f}%"), unsafe_allow_html=True)

        with col4:
            complete_cases = len(data.dropna())
            st.markdown(create_metric_card("Complete Cases", f"{complete_cases:,}"), unsafe_allow_html=True)

        st.markdown("---")

        # 数据分布图表
        col1, col2 = st.columns(2)

        with col1:
            if 'Basic_Demos_Enroll_Year' in data.columns:
                # 按年份分布
                year_dist = data['Basic_Demos_Enroll_Year'].value_counts().sort_index()
                fig_year = px.bar(
                    x=year_dist.index.tolist(),
                    y=year_dist.values.tolist(),
                    title="Participants by Enrollment Year",
                    labels={'x': 'Year', 'y': 'Count'},
                    color=year_dist.values.tolist(),
                    color_continuous_scale='viridis'
                )
                fig_year.update_layout(showlegend=False)
                st.plotly_chart(fig_year, use_container_width=True)

        with col2:
            if 'Basic_Demos_Study_Site' in data.columns:
                # 按研究站点分布
                site_dist = data['Basic_Demos_Study_Site'].value_counts().sort_index()
                fig_site = px.pie(
                    values=site_dist.values.tolist(),
                    names=[f"Site {i}" for i in site_dist.index],
                    title="Distribution by Study Site"
                )
                st.plotly_chart(fig_site, use_container_width=True)

        if 'ADHD_Outcome' in data.columns:
            st.subheader("Target Variables Distribution")
            adhd_counts = data['ADHD_Outcome'].value_counts()
            total_samples = len(data)
            non_adhd_count = adhd_counts.get(0, 0)
            adhd_count = adhd_counts.get(1, 0)
            non_adhd_pct = (non_adhd_count / total_samples) * 100
            adhd_pct = (adhd_count / total_samples) * 100

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"**Total samples: {total_samples}**")

            with col2:
                st.markdown(f"**Non-ADHD:** n={non_adhd_count} ({non_adhd_pct:.1f}%)")

            with col3:
                st.markdown(f"**ADHD:** n={adhd_count} ({adhd_pct:.1f}%)")

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

                # 添加ADHD诊断的环形图
                fig_adhd_donut = px.pie(
                    values=adhd_counts.values.tolist(),
                    names=['Non-ADHD', 'ADHD'],
                    title="Distribution of Type of Diagnosis",
                    color_discrete_map={'Non-ADHD': '#95a5a6', 'ADHD': '#f1c40f'},
                    hole=0.4
                )
                st.plotly_chart(fig_adhd_donut, use_container_width=True)

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

                # 添加性别的环形图
                fig_sex_donut = px.pie(
                    values=[sex_counts.get(0, 0), sex_counts.get(1, 0)],
                    names=['Male', 'Female'],
                    title="Distribution of Sex of participant",
                    color_discrete_map={'Male': '#74b9ff', 'Female': '#fd79a8'},
                    hole=0.4
                )
                st.plotly_chart(fig_sex_donut, use_container_width=True)

        # 变量类型分布
        st.subheader("Variable Types Distribution")

        # 数值变量和分类变量统计
        numeric_vars = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_vars = data.select_dtypes(include=['object', 'category']).columns.tolist()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(create_metric_card("Numeric Variables", f"{len(numeric_vars)}"), unsafe_allow_html=True)

        with col2:
            st.markdown(create_metric_card("Categorical Variables", f"{len(categorical_vars)}"), unsafe_allow_html=True)

        with col3:
            st.markdown(create_metric_card("Total Variables", f"{len(data.columns)}"), unsafe_allow_html=True)

    with tab2:
        # 相关性分析
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            st.subheader("Feature Correlations")

            # 选择要分析的特征
            selected_features = st.multiselect(
                "Select features for correlation analysis:",
                numeric_cols,
                default=numeric_cols[:8] if len(numeric_cols) >= 8 else numeric_cols
            )

            if len(selected_features) > 1:
                corr_matrix = data[selected_features].corr()

                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Matrix Heatmap",
                    color_continuous_scale="RdBu_r"
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)

            # 添加完整的相关性热力图
            st.subheader("Correlation Heatmap - All Variables")
            # 计算所有数值变量的相关性矩阵
            all_numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(all_numeric_cols) > 1:
                full_corr_matrix = data[all_numeric_cols].corr()

                # 创建下三角矩阵（只显示下三角部分）
                mask = np.triu(np.ones_like(full_corr_matrix, dtype=bool))
                corr_matrix_masked = full_corr_matrix.mask(mask)

                fig_full_corr = px.imshow(
                    corr_matrix_masked,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r"
                )
                fig_full_corr.update_layout(height=800)
                st.plotly_chart(fig_full_corr, use_container_width=True)

                # 添加性别相关性分析
                if 'Sex_F' in data.columns:
                    st.subheader("Gender Correlation Analysis")

                    # 计算与性别的相关性
                    sex_corr = full_corr_matrix['Sex_F'].sort_values(ascending=False)

                    # 显示性别相关性
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Top Positive Correlations with Gender:**")
                        st.dataframe(sex_corr.head(10), use_container_width=True)

                    with col2:
                        st.markdown("**Top Negative Correlations with Gender:**")
                        st.dataframe(sex_corr.tail(10), use_container_width=True)

        # 年龄分布分析
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
        # 缺失值分析
        missing_data = analyze_missing_values(data)

        if len(missing_data) > 0:
            st.subheader("Missing Values Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(missing_data, use_container_width=True)

            with col2:
                fig_missing = px.bar(
                    missing_data,
                    x='Missing_Percentage',
                    y='Feature',
                    orientation='h',
                    title="Missing Values by Feature",
                    color='Missing_Percentage',
                    color_continuous_scale='reds'
                )
                st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("🎉 No missing values detected in the dataset!")

        # 数据类型摘要
        st.subheader("Data Types Summary")

        dtype_summary = pd.DataFrame({
            'Data Type': [str(dtype) for dtype in data.dtypes.value_counts().index],
            'Count': data.dtypes.value_counts().values
        })

        fig_dtype = px.pie(
            dtype_summary,
            values='Count',
            names='Data Type',
            title="Distribution of Data Types"
        )
        st.plotly_chart(fig_dtype, use_container_width=True)

        # 唯一值统计
        st.subheader("Unique Values Statistics")

        unique_stats = pd.DataFrame({
            'Feature': data.columns,
            'Unique Values': data.nunique(),
            'Data Type': [str(dtype) for dtype in data.dtypes]
        }).sort_values('Unique Values', ascending=False)

        st.dataframe(unique_stats.head(15), use_container_width=True)

        # 异常值检测
        st.subheader("Outlier Detection")

        numeric_vars = data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_vars:
            selected_var = st.selectbox("Select variable for outlier detection:", numeric_vars)

            if selected_var:
                outliers, lower_bound, upper_bound = detect_outliers_iqr(data, selected_var)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Outlier Statistics for {selected_var}:**")
                    st.markdown(f"- Lower Bound: {lower_bound:.2f}")
                    st.markdown(f"- Upper Bound: {upper_bound:.2f}")
                    st.markdown(f"- Number of Outliers: {len(outliers)}")
                    st.markdown(f"- Outlier Percentage: {len(outliers) / len(data) * 100:.2f}%")

                with col2:
                    # 箱线图显示异常值
                    fig_box = px.box(
                        data,
                        y=selected_var,
                        title=f"Box Plot for {selected_var}",
                        points="outliers"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

    with tab4:
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