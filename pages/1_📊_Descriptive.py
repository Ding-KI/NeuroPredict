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

warnings.filterwarnings("ignore")

# 设置页面配置
st.set_page_config(
    page_title="Descriptive Analysis - NeuroPredict",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 0.5rem;
    }

    .metric-title {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }

    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
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

# 数据加载函数
@st.cache_data
def load_data():
    """加载数据"""
    try:
        # 尝试加载真实数据
        df_cat = pd.read_excel('data/raw_data/TRAIN/TRAIN_CATEGORICAL_METADATA_new.xlsx')
        df_Q = pd.read_excel('data/raw_data/TRAIN/TRAIN_QUANTITATIVE_METADATA_new.xlsx')
        df_sol = pd.read_excel('data/raw_data/TRAIN/TRAINING_SOLUTIONS.xlsx')

        overall_df = df_cat.merge(df_Q, on="participant_id", how="inner").merge(df_sol, on="participant_id", how="inner")
        
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

def perform_ttest(data, group_col, value_col, group1, group2):
    """执行t检验"""
    try:
        group1_data = data[data[group_col] == group1][value_col].dropna()
        group2_data = data[data[group_col] == group2][value_col].dropna()
        
        if len(group1_data) > 0 and len(group2_data) > 0:
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
            return t_stat, p_value, group1_data, group2_data
        else:
            return None, None, None, None
    except:
        return None, None, None, None

def main():
    # 主标题
    st.markdown('<h1 class="main-header">📊 Descriptive Analysis</h1>', unsafe_allow_html=True)
    
    # 加载数据
    data = load_data()
    
    # 侧边栏
    st.sidebar.title("📊 Descriptive Analysis")
    st.sidebar.markdown("---")
    
    # 数据集信息
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.info(f"""
    **Total Participants**: {len(data):,}  
    **Features**: {len(data.columns):,}  
    **Study Sites**: {data['Basic_Demos_Study_Site'].nunique()}  
    **Years**: {data['Basic_Demos_Enroll_Year'].nunique()}
    """)
    
    # 快速统计
    st.sidebar.markdown("### 🎯 Quick Stats")
    adhd_rate = (data['ADHD_Outcome'].sum() / len(data) * 100) if 'ADHD_Outcome' in data.columns else 0
    female_rate = (data['Sex_F'].sum() / len(data) * 100) if 'Sex_F' in data.columns else 0
    
    st.sidebar.metric("ADHD Rate", f"{adhd_rate:.1f}%")
    st.sidebar.metric("Female Participants", f"{female_rate:.1f}%")
    
    # 主要内容
    st.markdown("""
    <div class="info-box">
        <h3>🔍 Descriptive Analysis Overview</h3>
        <p>This section provides comprehensive descriptive statistics and visualizations to understand the distribution of participants across key demographics and their relationship with ADHD outcomes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Question 1: Demographics Distribution
    st.markdown('<div class="question-header">Question 1: Distribution of Participants by Key Demographics</div>', unsafe_allow_html=True)
    st.markdown("**What is the distribution of participants by key demographics (e.g., study site, ethnicity, race, sex)? Show counts and percentages in bar charts or tables.**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 研究站点分布
        if 'Basic_Demos_Study_Site' in data.columns:
            site_counts = data['Basic_Demos_Study_Site'].value_counts().sort_index()
            site_percentages = (site_counts / len(data) * 100).round(2)
            
            st.markdown("### 🏥 Study Site Distribution")
            site_df = pd.DataFrame({
                'Site': [f'Site {i}' for i in site_counts.index],
                'Count': site_counts.values,
                'Percentage': site_percentages.values
            })
            st.dataframe(site_df, use_container_width=True)
            
            # 研究站点饼图
            fig_site = px.pie(
                values=site_counts.values,
                names=[f"Site {i}" for i in site_counts.index],
                title="Study Site Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_site, use_container_width=True)
    
    with col2:
        # 性别分布
        if 'Gender' in data.columns:
            gender_counts = data['Gender'].value_counts()
            gender_percentages = (gender_counts / len(data) * 100).round(2)
            
            st.markdown("### 👥 Gender Distribution")
            gender_df = pd.DataFrame({
                'Gender': gender_counts.index,
                'Count': gender_counts.values,
                'Percentage': gender_percentages.values
            })
            st.dataframe(gender_df, use_container_width=True)
            
            # 性别柱状图
            fig_gender = px.bar(
                x=gender_counts.index,
                y=gender_counts.values,
                title="Gender Distribution",
                color=gender_counts.index,
                color_discrete_map={'Male': '#2ecc71', 'Female': '#9b59b6'}
            )
            st.plotly_chart(fig_gender, use_container_width=True)
    
    # 种族和民族分布
    col3, col4 = st.columns(2)
    
    with col3:
        if 'PreInt_Demos_Fam_Child_Ethnicity' in data.columns:
            ethnicity_counts = data['PreInt_Demos_Fam_Child_Ethnicity'].value_counts().sort_index()
            ethnicity_percentages = (ethnicity_counts / len(data) * 100).round(2)
            
            st.markdown("### 🌍 Ethnicity Distribution")
            ethnicity_labels = {0: 'Not Hispanic/Latino', 1: 'Hispanic/Latino', 2: 'Unknown', 3: 'Other'}
            ethnicity_df = pd.DataFrame({
                'Ethnicity': [ethnicity_labels.get(i, f'Category {i}') for i in ethnicity_counts.index],
                'Count': ethnicity_counts.values,
                'Percentage': ethnicity_percentages.values
            })
            st.dataframe(ethnicity_df, use_container_width=True)
    
    with col4:
        if 'PreInt_Demos_Fam_Child_Race' in data.columns:
            race_counts = data['PreInt_Demos_Fam_Child_Race'].value_counts().sort_index()
            race_percentages = (race_counts / len(data) * 100).round(2)
            
            st.markdown("### 🧬 Race Distribution")
            race_labels = {0: 'White', 1: 'Black/African American', 2: 'Asian', 3: 'American Indian/Alaska Native', 
                          4: 'Native Hawaiian/Pacific Islander', 5: 'Other', 6: 'Multi-racial', 7: 'Unknown', 
                          8: 'Refused', 9: 'Not Applicable', 10: 'Missing', 11: 'Other'}
            race_df = pd.DataFrame({
                'Race': [race_labels.get(i, f'Category {i}') for i in race_counts.index],
                'Count': race_counts.values,
                'Percentage': race_percentages.values
            })
            st.dataframe(race_df, use_container_width=True)
    
    # Question 2: ADHD Diagnosis Differences by Gender
    st.markdown('<div class="question-header">Question 2: ADHD Diagnosis Differences by Gender</div>', unsafe_allow_html=True)
    st.markdown("**Are there significant differences in ADHD diagnosis outcomes between males and females?**")
    
    if 'Gender' in data.columns and 'ADHD_Status' in data.columns:
        # 交叉表
        crosstab = pd.crosstab(data['Gender'], data['ADHD_Status'], margins=True)
        crosstab_pct = pd.crosstab(data['Gender'], data['ADHD_Status'], normalize='index') * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 ADHD Status by Gender (Counts)")
            st.dataframe(crosstab, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 ADHD Status by Gender (Percentages)")
            st.dataframe(crosstab_pct.round(2), use_container_width=True)
        
        # 统计检验
        chi2, p_value, contingency_table = perform_chi_square_test(data, 'Gender', 'ADHD_Status')
        
        if chi2 is not None:
            st.markdown("### 🔬 Statistical Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(create_metric_card("Chi-square", f"{chi2:.4f}"), unsafe_allow_html=True)
            
            with col2:
                st.markdown(create_metric_card("P-value", f"{p_value:.4f}"), unsafe_allow_html=True)
            
            with col3:
                significance = "Significant" if p_value < 0.05 else "Not Significant"
                st.markdown(create_metric_card("Result", significance), unsafe_allow_html=True)
            
            # 可视化
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
    
    # Question 3: APQ Questionnaire Distribution
    st.markdown('<div class="question-header">Question 3: APQ Questionnaire Distribution</div>', unsafe_allow_html=True)
    st.markdown("**How are the scores on each dimension of the APQ questionnaire distributed across different genders and ADHD diagnostic groups?**")
    
    # APQ变量
    apq_vars = [col for col in data.columns if col.startswith('APQ_P_APQ_P_')]
    
    if apq_vars and 'Gender' in data.columns and 'ADHD_Status' in data.columns:
        # 选择APQ变量
        selected_apq = st.selectbox("Select APQ Variable:", apq_vars)
        
        if selected_apq:
            # 描述性统计
            desc_stats = data.groupby(['Gender', 'ADHD_Status'])[selected_apq].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(2)
            
            st.markdown(f"### 📋 Descriptive Statistics for {selected_apq}")
            st.dataframe(desc_stats, use_container_width=True)
            
            # 可视化
            col1, col2 = st.columns(2)
            
            with col1:
                # 箱线图
                fig_box = px.box(
                    data,
                    x='Gender',
                    y=selected_apq,
                    color='ADHD_Status',
                    title=f"{selected_apq} Distribution by Gender and ADHD Status",
                    color_discrete_map={'ADHD': '#e74c3c', 'Non-ADHD': '#3498db'}
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                # 小提琴图
                fig_violin = px.violin(
                    data,
                    x='Gender',
                    y=selected_apq,
                    color='ADHD_Status',
                    title=f"{selected_apq} Distribution (Violin Plot)",
                    color_discrete_map={'ADHD': '#e74c3c', 'Non-ADHD': '#3498db'}
                )
                st.plotly_chart(fig_violin, use_container_width=True)
            
            # 统计检验
            st.markdown("### 🔬 Statistical Tests")
            
            # 按性别分组进行t检验
            male_adhd = data[(data['Gender'] == 'Male') & (data['ADHD_Status'] == 'ADHD')][selected_apq].dropna()
            male_non_adhd = data[(data['Gender'] == 'Male') & (data['ADHD_Status'] == 'Non-ADHD')][selected_apq].dropna()
            female_adhd = data[(data['Gender'] == 'Female') & (data['ADHD_Status'] == 'ADHD')][selected_apq].dropna()
            female_non_adhd = data[(data['Gender'] == 'Female') & (data['ADHD_Status'] == 'Non-ADHD')][selected_apq].dropna()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if len(male_adhd) > 0 and len(male_non_adhd) > 0:
                    t_stat_male, p_value_male = stats.ttest_ind(male_adhd, male_non_adhd)
                    st.markdown(f"**Male (ADHD vs Non-ADHD):**")
                    st.markdown(f"- t-statistic: {t_stat_male:.4f}")
                    st.markdown(f"- p-value: {p_value_male:.4f}")
                    st.markdown(f"- Significant: {'Yes' if p_value_male < 0.05 else 'No'}")
            
            with col2:
                if len(female_adhd) > 0 and len(female_non_adhd) > 0:
                    t_stat_female, p_value_female = stats.ttest_ind(female_adhd, female_non_adhd)
                    st.markdown(f"**Female (ADHD vs Non-ADHD):**")
                    st.markdown(f"- t-statistic: {t_stat_female:.4f}")
                    st.markdown(f"- p-value: {p_value_female:.4f}")
                    st.markdown(f"- Significant: {'Yes' if p_value_female < 0.05 else 'No'}")
    
    # Question 4: SDQ Questionnaire Distribution
    st.markdown('<div class="question-header">Question 4: SDQ Questionnaire Distribution</div>', unsafe_allow_html=True)
    st.markdown("**What are the distribution characteristics of the Difficulties Total, Externalizing, and Internalizing scores in the SDQ questionnaire across different genders and ADHD diagnostic groups?**")
    
    # SDQ变量
    sdq_vars = [col for col in data.columns if col.startswith('SDQ_SDQ_')]
    
    if sdq_vars and 'Gender' in data.columns and 'ADHD_Status' in data.columns:
        # 计算SDQ总分、外化问题和内化问题
        if 'SDQ_SDQ_Emotional_Problems' in data.columns and 'SDQ_SDQ_Hyperactivity' in data.columns and 'SDQ_SDQ_Conduct_Problems' in data.columns and 'SDQ_SDQ_Peer_Problems' in data.columns:
            data['SDQ_Total_Difficulties'] = data[['SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Conduct_Problems', 'SDQ_SDQ_Peer_Problems']].sum(axis=1)
            data['SDQ_Externalizing'] = data[['SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Conduct_Problems']].sum(axis=1)
            data['SDQ_Internalizing'] = data[['SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Peer_Problems']].sum(axis=1)
            
            # 选择SDQ变量
            sdq_analysis_vars = ['SDQ_Total_Difficulties', 'SDQ_Externalizing', 'SDQ_Internalizing']
            selected_sdq = st.selectbox("Select SDQ Variable:", sdq_analysis_vars)
            
            if selected_sdq:
                # 描述性统计
                desc_stats_sdq = data.groupby(['Gender', 'ADHD_Status'])[selected_sdq].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).round(2)
                
                st.markdown(f"### 📋 Descriptive Statistics for {selected_sdq}")
                st.dataframe(desc_stats_sdq, use_container_width=True)
                
                # 可视化
                col1, col2 = st.columns(2)
                
                with col1:
                    # 箱线图
                    fig_box_sdq = px.box(
                        data,
                        x='Gender',
                        y=selected_sdq,
                        color='ADHD_Status',
                        title=f"{selected_sdq} Distribution by Gender and ADHD Status",
                        color_discrete_map={'ADHD': '#e74c3c', 'Non-ADHD': '#3498db'}
                    )
                    st.plotly_chart(fig_box_sdq, use_container_width=True)
                
                with col2:
                    # 密度图
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
                
                # 统计检验
                st.markdown("### 🔬 Statistical Tests")
                
                # 按性别分组进行t检验
                male_adhd_sdq = data[(data['Gender'] == 'Male') & (data['ADHD_Status'] == 'ADHD')][selected_sdq].dropna()
                male_non_adhd_sdq = data[(data['Gender'] == 'Male') & (data['ADHD_Status'] == 'Non-ADHD')][selected_sdq].dropna()
                female_adhd_sdq = data[(data['Gender'] == 'Female') & (data['ADHD_Status'] == 'ADHD')][selected_sdq].dropna()
                female_non_adhd_sdq = data[(data['Gender'] == 'Female') & (data['ADHD_Status'] == 'Non-ADHD')][selected_sdq].dropna()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if len(male_adhd_sdq) > 0 and len(male_non_adhd_sdq) > 0:
                        t_stat_male_sdq, p_value_male_sdq = stats.ttest_ind(male_adhd_sdq, male_non_adhd_sdq)
                        st.markdown(f"**Male (ADHD vs Non-ADHD):**")
                        st.markdown(f"- t-statistic: {t_stat_male_sdq:.4f}")
                        st.markdown(f"- p-value: {p_value_male_sdq:.4f}")
                        st.markdown(f"- Significant: {'Yes' if p_value_male_sdq < 0.05 else 'No'}")
                
                with col2:
                    if len(female_adhd_sdq) > 0 and len(female_non_adhd_sdq) > 0:
                        t_stat_female_sdq, p_value_female_sdq = stats.ttest_ind(female_adhd_sdq, female_non_adhd_sdq)
                        st.markdown(f"**Female (ADHD vs Non-ADHD):**")
                        st.markdown(f"- t-statistic: {t_stat_female_sdq:.4f}")
                        st.markdown(f"- p-value: {p_value_female_sdq:.4f}")
                        st.markdown(f"- Significant: {'Yes' if p_value_female_sdq < 0.05 else 'No'}")
    
    # Question 5: Parents' Education and Occupation Distribution
    st.markdown('<div class="question-header">Question 5: Parents\' Education and Occupation Distribution</div>', unsafe_allow_html=True)
    st.markdown("**How are parents' education level and occupational status distributed among children with ADHD and those without ADHD?**")
    
    # 父母教育和职业变量
    parent_vars = {
        'Barratt_Barratt_P1_Edu': 'Parent 1 Education Level',
        'Barratt_Barratt_P2_Edu': 'Parent 2 Education Level', 
        'Barratt_Barratt_P1_Occ': 'Parent 1 Occupation Level',
        'Barratt_Barratt_P2_Occ': 'Parent 2 Occupation Level'
    }
    
    # 检查哪些变量存在
    available_parent_vars = {k: v for k, v in parent_vars.items() if k in data.columns}
    
    if available_parent_vars and 'ADHD_Status' in data.columns:
        # 选择要分析的变量
        selected_parent_var = st.selectbox("Select Parent Variable:", list(available_parent_vars.keys()), 
                                         format_func=lambda x: available_parent_vars[x])
        
        if selected_parent_var:
            # 描述性统计
            desc_stats_parent = data.groupby('ADHD_Status')[selected_parent_var].agg([
                'count', 'mean', 'std', 'min', 'max', 'median'
            ]).round(2)
            
            st.markdown(f"### 📋 Descriptive Statistics for {available_parent_vars[selected_parent_var]}")
            st.dataframe(desc_stats_parent, use_container_width=True)
            
            # 可视化
            col1, col2 = st.columns(2)
            
            with col1:
                # 箱线图
                fig_box_parent = px.box(
                    data,
                    x='ADHD_Status',
                    y=selected_parent_var,
                    title=f"{available_parent_vars[selected_parent_var]} Distribution by ADHD Status",
                    color='ADHD_Status',
                    color_discrete_map={'ADHD': '#e74c3c', 'Non-ADHD': '#3498db'}
                )
                st.plotly_chart(fig_box_parent, use_container_width=True)
            
            with col2:
                # 直方图
                fig_hist_parent = px.histogram(
                    data,
                    x=selected_parent_var,
                    color='ADHD_Status',
                    title=f"{available_parent_vars[selected_parent_var]} Distribution",
                    color_discrete_map={'ADHD': '#e74c3c', 'Non-ADHD': '#3498db'},
                    marginal='box',
                    nbins=20
                )
                st.plotly_chart(fig_hist_parent, use_container_width=True)
            
            # 统计检验
            st.markdown("### 🔬 Statistical Analysis")
            
            # 按ADHD状态分组进行t检验
            adhd_data = data[data['ADHD_Status'] == 'ADHD'][selected_parent_var].dropna()
            non_adhd_data = data[data['ADHD_Status'] == 'Non-ADHD'][selected_parent_var].dropna()
            
            if len(adhd_data) > 0 and len(non_adhd_data) > 0:
                t_stat, p_value = stats.ttest_ind(adhd_data, non_adhd_data)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(create_metric_card("T-statistic", f"{t_stat:.4f}"), unsafe_allow_html=True)
                
                with col2:
                    st.markdown(create_metric_card("P-value", f"{p_value:.4f}"), unsafe_allow_html=True)
                
                with col3:
                    significance = "Significant" if p_value < 0.05 else "Not Significant"
                    st.markdown(create_metric_card("Result", significance), unsafe_allow_html=True)
                
                # 效应大小 (Cohen's d)
                pooled_std = np.sqrt(((len(adhd_data) - 1) * adhd_data.std()**2 + 
                                    (len(non_adhd_data) - 1) * non_adhd_data.std()**2) / 
                                   (len(adhd_data) + len(non_adhd_data) - 2))
                cohens_d = (adhd_data.mean() - non_adhd_data.mean()) / pooled_std
                
                st.markdown(f"**Effect Size (Cohen's d):** {cohens_d:.4f}")
                
                # 解释效应大小
                if abs(cohens_d) < 0.2:
                    effect_interpretation = "Small effect"
                elif abs(cohens_d) < 0.5:
                    effect_interpretation = "Medium effect"
                elif abs(cohens_d) < 0.8:
                    effect_interpretation = "Large effect"
                else:
                    effect_interpretation = "Very large effect"
                
                st.markdown(f"**Effect Size Interpretation:** {effect_interpretation}")
            
            # 父母1和父母2的比较
            if 'Barratt_Barratt_P1_Edu' in data.columns and 'Barratt_Barratt_P2_Edu' in data.columns:
                st.markdown("### 👨‍👩‍👧‍👦 Parent 1 vs Parent 2 Comparison")
                
                # 教育水平比较
                if 'Barratt_Barratt_P1_Edu' in data.columns and 'Barratt_Barratt_P2_Edu' in data.columns:
                    edu_comparison = pd.DataFrame({
                        'Parent 1 Education': data['Barratt_Barratt_P1_Edu'].dropna(),
                        'Parent 2 Education': data['Barratt_Barratt_P2_Edu'].dropna()
                    })
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📚 Education Level Comparison")
                        edu_stats = edu_comparison.describe().round(2)
                        st.dataframe(edu_stats, use_container_width=True)
                        
                        # 教育水平相关性
                        edu_corr = edu_comparison.corr().iloc[0, 1]
                        st.markdown(f"**Education Correlation:** {edu_corr:.4f}")
                    
                    with col2:
                        # 教育水平散点图
                        fig_edu_scatter = px.scatter(
                            edu_comparison,
                            x='Parent 1 Education',
                            y='Parent 2 Education',
                            title='Parent 1 vs Parent 2 Education Level',
                            trendline='ols'
                        )
                        st.plotly_chart(fig_edu_scatter, use_container_width=True)
                
                # 职业水平比较
                if 'Barratt_Barratt_P1_Occ' in data.columns and 'Barratt_Barratt_P2_Occ' in data.columns:
                    occ_comparison = pd.DataFrame({
                        'Parent 1 Occupation': data['Barratt_Barratt_P1_Occ'].dropna(),
                        'Parent 2 Occupation': data['Barratt_Barratt_P2_Occ'].dropna()
                    })
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 💼 Occupation Level Comparison")
                        occ_stats = occ_comparison.describe().round(2)
                        st.dataframe(occ_stats, use_container_width=True)
                        
                        # 职业水平相关性
                        occ_corr = occ_comparison.corr().iloc[0, 1]
                        st.markdown(f"**Occupation Correlation:** {occ_corr:.4f}")
                    
                    with col2:
                        # 职业水平散点图
                        fig_occ_scatter = px.scatter(
                            occ_comparison,
                            x='Parent 1 Occupation',
                            y='Parent 2 Occupation',
                            title='Parent 1 vs Parent 2 Occupation Level',
                            trendline='ols'
                        )
                        st.plotly_chart(fig_occ_scatter, use_container_width=True)
    
    else:
        st.info("Parent education and occupation variables not available in the current dataset.")
    
    # 总结
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <h3>📊 Summary</h3>
        <p>This descriptive analysis provides a comprehensive overview of the dataset characteristics, including demographic distributions, ADHD diagnosis patterns by gender, questionnaire score distributions, and parent education/occupation patterns. The analysis includes both descriptive statistics and statistical tests to identify significant differences between groups.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

