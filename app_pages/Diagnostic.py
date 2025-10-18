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
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

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
    'Sex_F': 'Gender (Female)',
    # 新增的合成变量映射
    'SDQ_Total_Difficulties': 'Total Difficulties',
    'SDQ_Externalizing': 'Externalizing Behavior',
    'SDQ_Internalizing': 'Internalizing Behavior'
}

APQ_DIMENSIONS = {
            'APQ_P_APQ_P_CP': 'Corporal Punishment',
            'APQ_P_APQ_P_ID': 'Inconsistent Discipline',
            'APQ_P_APQ_P_INV': 'Parent Involvement',
            'APQ_P_APQ_P_OPD': 'Other Discipline Practices',
            'APQ_P_APQ_P_PM': 'Positive Monitoring',
            'APQ_P_APQ_P_PP': 'Positive Parenting'
        }

def map_feature_names(df, feature_column='Feature'):
    """
    将DataFrame中的特征名称映射为可读名称
    """
    df_copy = df.copy()
    df_copy[feature_column] = df_copy[feature_column].map(FEATURE_NAME_MAPPING).fillna(df_copy[feature_column])
    return df_copy

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

def calculate_apq_corr_p_matrices(data, apq_dimensions):
    vars_list = list(apq_dimensions.keys())
    n = len(vars_list)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))
    
    for i, var1 in enumerate(vars_list):
        for j, var2 in enumerate(vars_list):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
            else:
                mask = ~(data[var1].isna() | data[var2].isna())
                if mask.sum() > 2:  
                    corr, p = stats.pearsonr(data[var1][mask], data[var2][mask])
                    corr_matrix[i, j] = corr
                    p_matrix[i, j] = p
                else:
                    corr_matrix[i, j] = np.nan 
                    p_matrix[i, j] = np.nan
            
    return corr_matrix, p_matrix

def create_apq_correlation_heatmap(corr_matrix, p_matrix, title, apq_dimensions):
    labels = list(apq_dimensions.values())
    
    text_matrix = []
    for i in range(len(corr_matrix)):
        text_row = []
        for j in range(len(corr_matrix)):
            if i != j:
                text = f"{corr_matrix[i,j]:.2f}<br>p={p_matrix[i,j]:.3f}"
            else:
                text = "1.00"
            text_row.append(text)
        text_matrix.append(text_row)
    
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=labels,
        y=labels,
        zmin=-1, zmax=1,
        colorscale='RdBu', 
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        width=800,
        height=800,
        xaxis_tickangle=-45
    )
    
    return fig

def main():
    # 加载数据
    data = load_data()

    # 创建标签页
    tab1, tab2,tab3 = st.tabs([
        "Key Factors Analysis",
        "SDQ-APQ Relationships",
        "APQ Correlation Heatmap"
    ])

    with tab1:
        # Question 1: Key Factors Associated with ADHD
        st.markdown("### Key Factors Associated with ADHD Diagnosis")
        st.markdown(" * What are the key behavioral, parenting, and demographic factors associated with ADHD diagnosis?")
        st.markdown(" * How strong are these associations?")

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
                st.markdown("#### Feature Importance Analysis")

                importance_results, X_clean = perform_feature_importance_analysis(data, 'ADHD_Outcome', available_vars)

                if importance_results is not None:
                    # 显示前10个最重要的特征
                    top_features = importance_results.head(10)
                    
                    # 应用特征名称映射
                    top_features_mapped = map_feature_names(top_features)

                    st.markdown("##### Top 10 Most Important Features")
                        # F-score可视化
                    fig_fscore = px.bar(
                            top_features_mapped,
                            x='F_Score',
                            y='Feature',
                            orientation='h',
                            title='Feature Importance (F-Score)',
                            color='F_Score',
                            color_continuous_scale='viridis',
                            labels={'F_Score': 'Importance', 'Feature': 'Feature Name'}
                        )
                    # 使用与类别图表一致的高度计算
                    chart_height = max(400, 10 * 50)  # Top 10 特征使用固定间距
                    fig_fscore.update_layout(height=chart_height, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_fscore, use_container_width=True)

                    # 按类别分析
                    st.markdown("#### Analysis by Category")

                    categories = {
                        'Behavioral (SDQ)': [var for var in behavioral_vars if var in data.columns],
                        'Parenting (APQ)': [var for var in parenting_vars if var in data.columns],
                    }

                    for category, vars_in_category in categories.items():
                        if vars_in_category:
                            category_results = importance_results[importance_results['Feature'].isin(vars_in_category)]
                            if not category_results.empty:
                                st.markdown(f"##### {category}")

                                if len(category_results) > 1:
                                    # 应用特征名称映射
                                    category_results_mapped = map_feature_names(category_results)
                                    
                                    fig_category = px.bar(
                                        category_results_mapped,
                                        x='F_Score',
                                        y='Feature',
                                        orientation='h',
                                        title=f'{category} - Feature Importance',
                                        color='F_Score',
                                        color_continuous_scale='viridis',
                                        labels={'F_Score': 'Importance', 'Feature': 'Feature Name'}
                                    )
                                    # 根据特征数量调整图表高度，APQ使用与SDQ相同的间距
                                    if category == 'Parenting (APQ)':
                                        # APQ使用与SDQ相同的间距计算
                                        chart_height = max(400, 7 * 50)  # 使用SDQ的特征数量作为基准
                                    else:
                                        chart_height = max(300, len(category_results) * 50)
                                    fig_category.update_layout(height=chart_height)
                                    st.plotly_chart(fig_category, use_container_width=True)

    with tab2:
        # Question 3: SDQ-APQ Relationships by Gender
        st.markdown("### SDQ-APQ Relationships by Gender")
        st.markdown("* What is the relationship between SDQ subcategories (Externalizing, Internalizing, Total Difficulties) and APQ Parenting Practices?")
        st.markdown("* How does this relationship vary by gender?")

        # 计算SDQ子量表
        if 'SDQ_SDQ_Emotional_Problems' in data.columns and 'SDQ_SDQ_Hyperactivity' in data.columns and 'SDQ_SDQ_Conduct_Problems' in data.columns and 'SDQ_SDQ_Peer_Problems' in data.columns:
            data['SDQ_Externalizing'] = data[['SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Conduct_Problems']].sum(axis=1)
            data['SDQ_Internalizing'] = data[['SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Peer_Problems']].sum(axis=1)
            data['SDQ_Total_Difficulties'] = data[
                ['SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Conduct_Problems',
                 'SDQ_SDQ_Peer_Problems']].sum(axis=1)

            sdq_vars = ['SDQ_Externalizing', 'SDQ_Internalizing', 'SDQ_Total_Difficulties']
            apq_vars = [col for col in data.columns if col.startswith('APQ_P_APQ_P_')]

            if sdq_vars and apq_vars and 'Gender' in data.columns:
                st.markdown("#### SDQ-APQ Relationship Analysis")

                # 选择要分析的变量
                col1, col2 = st.columns(2)

                with col1:
                    # SDQ选择器 - 使用可读名称
                    sdq_readable_names = [FEATURE_NAME_MAPPING.get(var, var) for var in sdq_vars]
                    sdq_var_mapping = dict(zip(sdq_readable_names, sdq_vars))
                    selected_sdq_readable = st.selectbox("Select SDQ Variable:", sdq_readable_names)
                    selected_sdq = sdq_var_mapping[selected_sdq_readable]

                with col2:
                    # APQ选择器 - 使用可读名称
                    apq_readable_names = [FEATURE_NAME_MAPPING.get(var, var) for var in apq_vars]
                    apq_var_mapping = dict(zip(apq_readable_names, apq_vars))
                    selected_apq_readable = st.selectbox("Select APQ Variable:", apq_readable_names)
                    selected_apq = apq_var_mapping[selected_apq_readable]

                if selected_sdq and selected_apq:
                    # 散点图分析
                    st.markdown("#### Scatter Plot Analysis by Gender")

                    fig_scatter = px.scatter(
                        data,
                        x=selected_apq,
                        y=selected_sdq,
                        color='Gender',
                        title=f'{selected_sdq_readable} vs {selected_apq_readable} - Relationship by Gender',
                        trendline='ols',
                        color_discrete_map={'Male': '#2ecc71', 'Female': '#9b59b6'},
                        labels={
                            selected_apq: selected_apq_readable,
                            selected_sdq: selected_sdq_readable
                        }
                    )
                    fig_scatter.update_layout(height=500)
                    st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        st.markdown("### APQ Correlation Heatmap")
        st.markdown("How do different parenting practices (measured by APQ) correlate with each other, and do these correlations differ between ADHD and non-ADHD groups?")
        st.markdown("#### Correlation Heatmaps by Group")
        apq_vars_list = list(APQ_DIMENSIONS.keys())
        if all(col in data.columns for col in apq_vars_list) and 'ADHD_Outcome' in data.columns:
            
            # --- 数据计算 ---
            # @st.cache_data 似乎不适用于在st.tabs内部计算，因此我们直接计算
            # 1. 总体
            overall_corr, overall_p = calculate_apq_corr_p_matrices(data, APQ_DIMENSIONS)
            
            # 2. ADHD 组
            adhd_data = data[data['ADHD_Outcome'] == 1]
            adhd_corr, adhd_p = calculate_apq_corr_p_matrices(adhd_data, APQ_DIMENSIONS)
            
            # 3. Non-ADHD 组
            non_adhd_data = data[data['ADHD_Outcome'] == 0]
            non_adhd_corr, non_adhd_p = calculate_apq_corr_p_matrices(non_adhd_data, APQ_DIMENSIONS)

            # --- 界面 ---
            heatmap_choice = st.selectbox(
                "Select group to view correlation heatmap:",
                ("Overall Sample", "ADHD Group", "Non-ADHD Group")
            )

            if heatmap_choice == "Overall Sample":
                fig = create_apq_correlation_heatmap(
                    overall_corr, overall_p, 
                    "APQ Correlations - Overall Sample", 
                    APQ_DIMENSIONS
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif heatmap_choice == "ADHD Group":
                fig = create_apq_correlation_heatmap(
                    adhd_corr, adhd_p, 
                    "APQ Correlations - ADHD Group", 
                    APQ_DIMENSIONS
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif heatmap_choice == "Non-ADHD Group":
                fig = create_apq_correlation_heatmap(
                    non_adhd_corr, non_adhd_p, 
                    "APQ Correlations - Non-ADHD Group", 
                    APQ_DIMENSIONS
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Required APQ (APQ_P_APQ_P_*) or ADHD_Outcome columns not found in the data.")

        with st.expander("Show Key Findings Summary"):
            st.markdown("""
            **Overall Sample:**
            - Disciplinary dimensions clustered together. Corporal punishment was positively correlated with inconsistent discipline ($r = 0.244, p < 0.001$) and other discipline practices ($r = 0.295, p < 0.001$).
            - Inconsistent discipline also correlated positively with other discipline practices ($r = 0.284, p < 0.001$) and with positive monitoring ($r = 0.206, p < 0.001$), while showing a negative correlation with parent involvement ($r = -0.213, p < 0.001$).
            - Parent involvement had a strong positive correlation with positive parenting ($r = 0.563, p < 0.001$) and a negative correlation with positive monitoring ($r = -0.211, p < 0.001$).

            **Comparison between ADHD and Non-ADHD Groups:**
            - The correlation between **corporal punishment and inconsistent discipline** was slightly stronger in the ADHD group ($r = 0.257$) than in the non-ADHD group ($r = 0.192$).
            - Inconsistent discipline showed a **weaker negative association with parent involvement** in ADHD families ($r = -0.180$ vs. $r = -0.256$).
            - Some correlations were weaker in ADHD households, including **inconsistent discipline with other discipline practices** ($r = 0.242$ vs. $r = 0.336$).
            - Conversely, the correlation between **parent involvement and positive parenting** was stronger in the ADHD group ($r = 0.591$) than in non-ADHD participants ($r = 0.504$).

            **Conclusion:**
            Overall, ADHD participants exhibited stronger alignment between parent involvement and positive parenting, while clustering of negative disciplinary practices was slightly weaker than in non-ADHD participants.
            """)
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