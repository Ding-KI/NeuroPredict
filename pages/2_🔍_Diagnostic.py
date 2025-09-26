import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings

warnings.filterwarnings("ignore")

# 设置页面配置
st.set_page_config(
    page_title="Diagnostic Analysis - NeuroPredict",
    page_icon="🔍",
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

    .correlation-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
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

def create_metric_card(title, value, delta=None):
    """创建自定义指标卡片"""
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """

def calculate_correlation_matrix(data, variables, method='pearson'):
    """计算相关性矩阵"""
    corr_data = data[variables].dropna()
    if method == 'pearson':
        corr_matrix = corr_data.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = corr_data.corr(method='spearman')
    return corr_matrix

def perform_feature_importance_analysis(data, target_col, feature_cols):
    """执行特征重要性分析"""
    # 准备数据
    X = data[feature_cols].fillna(data[feature_cols].median())
    y = data[target_col]
    
    # 移除缺失值
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X_clean = X[mask]
    y_clean = y[mask]
    
    if len(X_clean) == 0:
        return None, None
    
    # F-test特征选择
    f_scores, f_pvalues = f_classif(X_clean, y_clean)
    
    # 互信息特征选择
    mi_scores = mutual_info_classif(X_clean, y_clean, random_state=42)
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        'Feature': feature_cols,
        'F_Score': f_scores,
        'F_P_Value': f_pvalues,
        'Mutual_Info': mi_scores
    })
    
    # 排序
    results = results.sort_values('F_Score', ascending=False)
    
    return results, X_clean

def main():
    # 主标题
    st.markdown('<h1 class="main-header">🔍 Diagnostic Analysis</h1>', unsafe_allow_html=True)
    
    # 加载数据
    data = load_data()
    
    # 侧边栏
    st.sidebar.title("🔍 Diagnostic Analysis")
    st.sidebar.markdown("---")
    
    # 数据集信息
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.info(f"""
    **Total Participants**: {len(data):,}  
    **Features**: {len(data.columns):,}  
    **ADHD Cases**: {data['ADHD_Outcome'].sum() if 'ADHD_Outcome' in data.columns else 0:,}  
    **Non-ADHD Cases**: {len(data) - data['ADHD_Outcome'].sum() if 'ADHD_Outcome' in data.columns else 0:,}
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
        <h3>🔍 Diagnostic Analysis Overview</h3>
        <p>This section provides in-depth diagnostic analysis to understand the relationships between behavioral, parenting, and demographic factors with ADHD diagnosis, and explores correlations between different measures.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Question 1: Key Factors Associated with ADHD
    st.markdown('<div class="question-header">Question 1: Key Factors Associated with ADHD Diagnosis</div>', unsafe_allow_html=True)
    st.markdown("**What are the key behavioral, parenting, and demographic factors associated with ADHD diagnosis, and how strong are these associations?**")
    
    if 'ADHD_Outcome' in data.columns:
        # 定义分析变量
        behavioral_vars = [col for col in data.columns if col.startswith('SDQ_')]
        parenting_vars = [col for col in data.columns if col.startswith('APQ_')]
        demographic_vars = ['Sex_F', 'MRI_Track_Age_at_Scan', 'Basic_Demos_Study_Site', 
                           'PreInt_Demos_Fam_Child_Ethnicity', 'PreInt_Demos_Fam_Child_Race']
        parent_edu_occ_vars = [col for col in data.columns if 'Barratt' in col]
        
        all_feature_vars = behavioral_vars + parenting_vars + demographic_vars + parent_edu_occ_vars
        # 过滤存在的变量
        available_vars = [var for var in all_feature_vars if var in data.columns]
        
        if available_vars:
            # 特征重要性分析
            st.markdown("### 🎯 Feature Importance Analysis")
            
            importance_results, X_clean = perform_feature_importance_analysis(data, 'ADHD_Outcome', available_vars)
            
            if importance_results is not None:
                # 显示前10个最重要的特征
                top_features = importance_results.head(10)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Top 10 Most Important Features")
                    st.dataframe(top_features, use_container_width=True)
                
                with col2:
                    # F-score可视化
                    fig_fscore = px.bar(
                        top_features,
                        x='F_Score',
                        y='Feature',
                        orientation='h',
                        title='Feature Importance (F-Score)',
                        color='F_Score',
                        color_continuous_scale='viridis'
                    )
                    fig_fscore.update_layout(height=400)
                    st.plotly_chart(fig_fscore, use_container_width=True)
                
                # 互信息可视化
                st.markdown("#### 🔗 Mutual Information Analysis")
                fig_mi = px.bar(
                    top_features,
                    x='Mutual_Info',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance (Mutual Information)',
                    color='Mutual_Info',
                    color_continuous_scale='plasma'
                )
                fig_mi.update_layout(height=400)
                st.plotly_chart(fig_mi, use_container_width=True)
                
                # 按类别分析
                st.markdown("### 📋 Analysis by Category")
                
                categories = {
                    'Behavioral (SDQ)': [var for var in behavioral_vars if var in data.columns],
                    'Parenting (APQ)': [var for var in parenting_vars if var in data.columns],
                    'Demographics': [var for var in demographic_vars if var in data.columns],
                    'Parent Education/Occupation': [var for var in parent_edu_occ_vars if var in data.columns]
                }
                
                for category, vars_in_category in categories.items():
                    if vars_in_category:
                        category_results = importance_results[importance_results['Feature'].isin(vars_in_category)]
                        if not category_results.empty:
                            st.markdown(f"#### {category}")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.dataframe(category_results, use_container_width=True)
                            
                            with col2:
                                if len(category_results) > 1:
                                    fig_category = px.bar(
                                        category_results,
                                        x='F_Score',
                                        y='Feature',
                                        orientation='h',
                                        title=f'{category} - Feature Importance',
                                        color='F_Score',
                                        color_continuous_scale='viridis'
                                    )
                                    st.plotly_chart(fig_category, use_container_width=True)
    
    # Question 2: APQ Correlations
    st.markdown('<div class="question-header">Question 2: APQ Parenting Practices Correlations</div>', unsafe_allow_html=True)
    st.markdown("**How do different parenting practices (measured by APQ) correlate with each other, and do these correlations differ between ADHD and non-ADHD groups?**")
    
    # APQ变量
    apq_vars = [col for col in data.columns if col.startswith('APQ_P_APQ_P_')]
    
    if apq_vars and 'ADHD_Status' in data.columns:
        st.markdown("### 🔗 APQ Correlation Analysis")
        
        # 整体相关性矩阵
        st.markdown("#### 📊 Overall APQ Correlation Matrix")
        corr_matrix = calculate_correlation_matrix(data, apq_vars)
        
        fig_corr_overall = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="APQ Variables Correlation Matrix (Overall)",
            color_continuous_scale="RdBu_r"
        )
        fig_corr_overall.update_layout(height=500)
        st.plotly_chart(fig_corr_overall, use_container_width=True)
        
        # 按ADHD状态分组的相关性分析
        st.markdown("#### 🎯 APQ Correlations by ADHD Status")
        
        adhd_data = data[data['ADHD_Status'] == 'ADHD'][apq_vars].dropna()
        non_adhd_data = data[data['ADHD_Status'] == 'Non-ADHD'][apq_vars].dropna()
        
        if len(adhd_data) > 0 and len(non_adhd_data) > 0:
            corr_adhd = calculate_correlation_matrix(data[data['ADHD_Status'] == 'ADHD'], apq_vars)
            corr_non_adhd = calculate_correlation_matrix(data[data['ADHD_Status'] == 'Non-ADHD'], apq_vars)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_corr_adhd = px.imshow(
                    corr_adhd,
                    text_auto=True,
                    aspect="auto",
                    title="APQ Correlations - ADHD Group",
                    color_continuous_scale="RdBu_r"
                )
                fig_corr_adhd.update_layout(height=400)
                st.plotly_chart(fig_corr_adhd, use_container_width=True)
            
            with col2:
                fig_corr_non_adhd = px.imshow(
                    corr_non_adhd,
                    text_auto=True,
                    aspect="auto",
                    title="APQ Correlations - Non-ADHD Group",
                    color_continuous_scale="RdBu_r"
                )
                fig_corr_non_adhd.update_layout(height=400)
                st.plotly_chart(fig_corr_non_adhd, use_container_width=True)
            
            # 相关性差异分析
            st.markdown("#### 📈 Correlation Differences Between Groups")
            
            # 计算相关性差异
            corr_diff = corr_adhd - corr_non_adhd
            
            fig_corr_diff = px.imshow(
                corr_diff,
                text_auto=True,
                aspect="auto",
                title="Correlation Differences (ADHD - Non-ADHD)",
                color_continuous_scale="RdBu_r"
            )
            fig_corr_diff.update_layout(height=500)
            st.plotly_chart(fig_corr_diff, use_container_width=True)
            
            # 显示最大的相关性差异
            st.markdown("#### 🔍 Largest Correlation Differences")
            
            # 获取上三角矩阵的差异
            mask = np.triu(np.ones_like(corr_diff, dtype=bool), k=1)
            corr_diff_masked = corr_diff.mask(mask)
            
            # 展平并排序
            corr_diff_flat = corr_diff_masked.stack().reset_index()
            corr_diff_flat.columns = ['Variable1', 'Variable2', 'Correlation_Difference']
            corr_diff_flat['Abs_Difference'] = abs(corr_diff_flat['Correlation_Difference'])
            corr_diff_flat = corr_diff_flat.sort_values('Abs_Difference', ascending=False)
            
            st.dataframe(corr_diff_flat.head(10), use_container_width=True)
    
    # Question 3: SDQ-APQ Relationships by Gender
    st.markdown('<div class="question-header">Question 3: SDQ-APQ Relationships by Gender</div>', unsafe_allow_html=True)
    st.markdown("**What is the relationship between SDQ subscales (Externalizing, Internalizing, Total Difficulties) and parenting practices (APQ dimensions), and does this relationship vary by gender?**")
    
    # 计算SDQ子量表
    if 'SDQ_SDQ_Emotional_Problems' in data.columns and 'SDQ_SDQ_Hyperactivity' in data.columns and 'SDQ_SDQ_Conduct_Problems' in data.columns and 'SDQ_SDQ_Peer_Problems' in data.columns:
        data['SDQ_Externalizing'] = data[['SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Conduct_Problems']].sum(axis=1)
        data['SDQ_Internalizing'] = data[['SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Peer_Problems']].sum(axis=1)
        data['SDQ_Total_Difficulties'] = data[['SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Conduct_Problems', 'SDQ_SDQ_Peer_Problems']].sum(axis=1)
        
        sdq_vars = ['SDQ_Externalizing', 'SDQ_Internalizing', 'SDQ_Total_Difficulties']
        apq_vars = [col for col in data.columns if col.startswith('APQ_P_APQ_P_')]
        
        if sdq_vars and apq_vars and 'Gender' in data.columns:
            st.markdown("### 🔗 SDQ-APQ Relationship Analysis")
            
            # 选择要分析的变量
            col1, col2 = st.columns(2)
            
            with col1:
                selected_sdq = st.selectbox("Select SDQ Variable:", sdq_vars)
            
            with col2:
                selected_apq = st.selectbox("Select APQ Variable:", apq_vars)
            
            if selected_sdq and selected_apq:
                # 整体相关性
                st.markdown("#### 📊 Overall SDQ-APQ Correlation")
                
                # 确保两个变量有相同的有效数据点
                valid_data = data[[selected_sdq, selected_apq]].dropna()
                if len(valid_data) > 0:
                    overall_corr, overall_p = pearsonr(valid_data[selected_sdq], valid_data[selected_apq])
                else:
                    overall_corr, overall_p = np.nan, np.nan
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if not np.isnan(overall_corr):
                        st.markdown(create_metric_card("Correlation", f"{overall_corr:.4f}"), unsafe_allow_html=True)
                    else:
                        st.markdown(create_metric_card("Correlation", "N/A"), unsafe_allow_html=True)
                
                with col2:
                    if not np.isnan(overall_p):
                        st.markdown(create_metric_card("P-value", f"{overall_p:.4f}"), unsafe_allow_html=True)
                    else:
                        st.markdown(create_metric_card("P-value", "N/A"), unsafe_allow_html=True)
                
                with col3:
                    if not np.isnan(overall_p):
                        significance = "Significant" if overall_p < 0.05 else "Not Significant"
                    else:
                        significance = "N/A"
                    st.markdown(create_metric_card("Result", significance), unsafe_allow_html=True)
                
                # 按性别分组分析
                st.markdown("#### 👥 SDQ-APQ Correlations by Gender")
                
                gender_correlations = []
                
                for gender in ['Male', 'Female']:
                    gender_data = data[data['Gender'] == gender]
                    if len(gender_data) > 10:  # 确保有足够的数据点
                        valid_gender_data = gender_data[[selected_sdq, selected_apq]].dropna()
                        if len(valid_gender_data) > 10:
                            corr, p_val = pearsonr(valid_gender_data[selected_sdq], valid_gender_data[selected_apq])
                            gender_correlations.append({
                                'Gender': gender,
                                'Correlation': corr,
                                'P_Value': p_val,
                                'Sample_Size': len(valid_gender_data)
                            })
                
                if gender_correlations:
                    gender_corr_df = pd.DataFrame(gender_correlations)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(gender_corr_df, use_container_width=True)
                    
                    with col2:
                        # 相关性比较图
                        fig_gender_corr = px.bar(
                            gender_corr_df,
                            x='Gender',
                            y='Correlation',
                            title=f'{selected_sdq} vs {selected_apq} - Correlation by Gender',
                            color='Correlation',
                            color_continuous_scale='RdBu_r'
                        )
                        st.plotly_chart(fig_gender_corr, use_container_width=True)
                
                # 散点图分析
                st.markdown("#### 📈 Scatter Plot Analysis by Gender")
                
                fig_scatter = px.scatter(
                    data,
                    x=selected_apq,
                    y=selected_sdq,
                    color='Gender',
                    title=f'{selected_sdq} vs {selected_apq} - Relationship by Gender',
                    trendline='ols',
                    color_discrete_map={'Male': '#2ecc71', 'Female': '#9b59b6'}
                )
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # 按性别和ADHD状态分析
                if 'ADHD_Status' in data.columns:
                    st.markdown("#### 🎯 SDQ-APQ Correlations by Gender and ADHD Status")
                    
                    # 创建分组分析
                    group_correlations = []
                    
                    for gender in ['Male', 'Female']:
                        for adhd_status in ['ADHD', 'Non-ADHD']:
                            group_data = data[(data['Gender'] == gender) & (data['ADHD_Status'] == adhd_status)]
                            if len(group_data) > 10:
                                valid_group_data = group_data[[selected_sdq, selected_apq]].dropna()
                                if len(valid_group_data) > 10:
                                    corr, p_val = pearsonr(valid_group_data[selected_sdq], valid_group_data[selected_apq])
                                    group_correlations.append({
                                        'Gender': gender,
                                        'ADHD_Status': adhd_status,
                                        'Correlation': corr,
                                        'P_Value': p_val,
                                        'Sample_Size': len(valid_group_data)
                                    })
                    
                    if group_correlations:
                        group_corr_df = pd.DataFrame(group_correlations)
                        
                        # 热力图
                        pivot_corr = group_corr_df.pivot(index='Gender', columns='ADHD_Status', values='Correlation')
                        
                        fig_heatmap = px.imshow(
                            pivot_corr,
                            text_auto=True,
                            aspect="auto",
                            title=f'{selected_sdq} vs {selected_apq} - Correlation Heatmap',
                            color_continuous_scale="RdBu_r"
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        # 详细表格
                        st.dataframe(group_corr_df, use_container_width=True)
    
    # 总结
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <h3>🔍 Diagnostic Analysis Summary</h3>
        <p>This diagnostic analysis provides comprehensive insights into the relationships between behavioral, parenting, and demographic factors with ADHD diagnosis. Key findings include feature importance rankings, correlation patterns between parenting practices, and gender-specific relationships between behavioral measures and parenting styles.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
