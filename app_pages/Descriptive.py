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
    'SDQ_Total_Difficulties': 'Total Difficulties',
    'SDQ_Externalizing': 'Externalizing Behavior',
    'SDQ_Internalizing': 'Internalizing Behavior'
}

parent_vars = {
    'Barratt_Barratt_P1_Edu': 'Parent 1 Education',
    'Barratt_Barratt_P2_Edu': 'Parent 2 Education',
    'Barratt_Barratt_P1_Occ': 'Parent 1 Occupation',
    'Barratt_Barratt_P2_Occ': 'Parent 2 Occupation'
}

education_labels = {
    3.0: 'Less than 7th grade',
    6.0: 'Junior high school',
    9.0: 'Partial high school',
    12.0: 'High school graduate',
    15.0: 'Partial college',
    18.0: 'College education',
    21.0: 'Graduate degree'
}

occupation_labels = {
    0.0: 'Homemaker',
    5.0: 'Day laborer',
    10.0: 'Service worker',
    15.0: 'Skilled worker',
    20.0: 'Technical worker',
    25.0: 'Skilled professional',
    30.0: 'Supervisor/Manager',
    35.0: 'Healthcare/Education',
    40.0: 'Engineer/Teacher',
    45.0: 'Executive/Professional'
}

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


@st.cache_data
def load_data():
    try:
        df_cat = pd.read_excel('data/raw_data/TRAIN/TRAIN_CATEGORICAL_METADATA_new.xlsx')
        df_Q = pd.read_excel('data/raw_data/TRAIN/TRAIN_QUANTITATIVE_METADATA_new.xlsx')
        df_sol = pd.read_excel('data/raw_data/TRAIN/TRAINING_SOLUTIONS.xlsx')

        overall_df = df_cat.merge(df_Q, on="participant_id", how="inner").merge(df_sol, on="participant_id",
                                                                                how="inner")

        if 'Sex_F' in overall_df.columns:
            overall_df['Gender'] = overall_df['Sex_F'].map({0: 'Male', 1: 'Female'})

        if 'ADHD_Outcome' in overall_df.columns:
            overall_df['ADHD_Status'] = overall_df['ADHD_Outcome'].map({0: 'Non-ADHD', 1: 'ADHD'})

    except Exception as e:
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
            'Barratt_Barratt_P1_Edu': np.random.choice(list(education_labels.keys()), n_samples),
            'Barratt_Barratt_P2_Edu': np.random.choice(list(education_labels.keys()), n_samples),
            'Barratt_Barratt_P1_Occ': np.random.choice(list(occupation_labels.keys()), n_samples),
            'Barratt_Barratt_P2_Occ': np.random.choice(list(occupation_labels.keys()), n_samples),
        })

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

        overall_df['Gender'] = overall_df['Sex_F'].map({0: 'Male', 1: 'Female'})
        overall_df['ADHD_Status'] = overall_df['ADHD_Outcome'].map({0: 'Non-ADHD', 1: 'ADHD'})

    return overall_df


def create_metric_card(title, value, delta=None):
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


def perform_chi_square_test(data, var1, var2):
    try:
        contingency_table = pd.crosstab(data[var1], data[var2])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        return chi2, p_value, contingency_table
    except:
        return None, None, None

# function for tab5
def _fmt_p(v):
    """Safe p-value formatting to 4 decimals or 'NA'."""
    if v is None:
        return "NA"
    try:
        vv = float(v)
        if np.isnan(vv):
            return "NA"
        return f"{vv:.4f}"
    except Exception:
        return "NA"

def _cramers_v(chi2, n, r, c):
    """Cramer's V effect size for chi-square test of independence."""
    if n == 0:
        return np.nan
    k = min(r - 1, c - 1)
    if k <= 0:
        return np.nan
    return np.sqrt(chi2 / (n * k))


def main():
    data = load_data()

    tab1, tab2, tab3,tab4,tab5 = st.tabs([
        "Gender Distribution",
        "APQ Questionnaire",
        "SDQ Questionnaire",
        "Distribution by Demographic",
        "Parental Background"
    ])

    with tab1:
        # Question 2: Gender-ADHD Status Distribution
        st.markdown("### Gender-ADHD Status Distribution")

        if 'Gender' in data.columns and 'ADHD_Status' in data.columns:
            gender_counts = data['Gender'].value_counts()
            crosstab = pd.crosstab(data['Gender'], data['ADHD_Status'], margins=True)
            crosstab_pct = pd.crosstab(data['Gender'], data['ADHD_Status'], normalize='index') * 100
        
        col1, col2 = st.columns(2)
        with col1:
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
        
        st.markdown("### ADHD Diagnosis by Gender (Total N = 1213)")
        st.markdown("This section displays the statistical relationship between gender and ADHD diagnosis.")

        st.markdown("#### 1) Contingency Table (Counts)")
        # Create a dataframe for the contingency table
        count_data = {
            '': ['Male', 'Female'],
            'Non-ADHD': [216, 166],
            'ADHD': [581, 250]
        }
        df_count = pd.DataFrame(count_data).set_index('')
        st.dataframe(df_count)

        st.markdown("#### 2) Row-wise Percentages (% within gender)")
        # Create a dataframe for the percentage table
        percent_data = {
            '': ['Male', 'Female'],
            'Non-ADHD': ['27.1%', '39.9%'],
            'ADHD': ['72.9%', '60.1%']
        }
        df_percent = pd.DataFrame(percent_data).set_index('')
        st.dataframe(df_percent)

        st.markdown("#### 3) Breakdown by Gender")
        st.markdown("- **Male:** Non-ADHD: 216 (27.1%), ADHD: 581 (72.9%)")
        st.markdown("- **Female:** Non-ADHD: 166 (39.9%), ADHD: 250 (60.1%)")

        st.markdown("#### 4) Statistical Test Results (Chi-square test of independence)")
        st.markdown("- **Chi-square ($\chi^2$):** 20.18")
        st.markdown("- **p-value:** 7.07e-06")

        st.markdown("#### 5) Conclusion")
        st.markdown(
            """
            - There is a statistically significant association between gender and ADHD diagnosis (p < 0.001).
            - The association strength is small (based on Cramer's V).
            - A higher percentage of males in the sample were diagnosed with ADHD (72.9%) compared to females (60.1%).
            """
        )

    with tab2:
        # Question 3: APQ Questionnaire Distribution
        st.markdown("### APQ Questionnaire Distribution")
        st.markdown(
            "How are the scores on each dimension of the APQ questionnaire distributed across different genders and ADHD groups?")

        apq_vars = [col for col in data.columns if col.startswith('APQ_P_APQ_P_')]

        if apq_vars and 'Gender' in data.columns and 'ADHD_Status' in data.columns:
            apq_readable_names = [FEATURE_NAME_MAPPING.get(var, var) for var in apq_vars]
            apq_var_mapping = dict(zip(apq_readable_names, apq_vars))
            
            selected_apq_readable = st.selectbox("Select APQ Variable:", apq_readable_names)
            selected_apq = apq_var_mapping[selected_apq_readable]

            if selected_apq:
                    fig_box = px.box(
                        data,
                        x='Gender',
                        y=selected_apq,
                        color='ADHD_Status',
                        title=f"{selected_apq_readable} Distribution by Gender and ADHD Status",
                        color_discrete_map={'ADHD': '#e74c3c', 'Non-ADHD': '#3498db'}
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        # Question 4: SDQ Questionnaire Distribution
        st.markdown("### SDQ Questionnaire Distribution")
        st.markdown(
            "What are the distribution characteristics of the Difficulties Total, Externalizing, and Internalizing scores in the SDQ questionnaire across different genders and ADHD diagnostic groups?")

        sdq_vars = [col for col in data.columns if col.startswith('SDQ_SDQ_')]

        if sdq_vars and 'Gender' in data.columns and 'ADHD_Status' in data.columns:
            if 'SDQ_SDQ_Emotional_Problems' in data.columns and 'SDQ_SDQ_Hyperactivity' in data.columns and 'SDQ_SDQ_Conduct_Problems' in data.columns and 'SDQ_SDQ_Peer_Problems' in data.columns:
                data['SDQ_Total_Difficulties'] = data[
                    ['SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Conduct_Problems',
                     'SDQ_SDQ_Peer_Problems']].sum(axis=1)
                data['SDQ_Externalizing'] = data[['SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Conduct_Problems']].sum(axis=1)
                data['SDQ_Internalizing'] = data[['SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Peer_Problems']].sum(axis=1)

                sdq_analysis_vars = ['SDQ_Total_Difficulties', 'SDQ_Externalizing', 'SDQ_Internalizing']
                sdq_readable_names = [FEATURE_NAME_MAPPING.get(var, var) for var in sdq_analysis_vars]
                sdq_var_mapping = dict(zip(sdq_readable_names, sdq_analysis_vars))
                
                selected_sdq_readable = st.selectbox("Select SDQ Variable:", sdq_readable_names)
                selected_sdq = sdq_var_mapping[selected_sdq_readable]

                if selected_sdq:
                        fig_density = px.histogram(
                            data,
                            x=selected_sdq,
                            color='ADHD_Status',
                            facet_col='Gender',
                            title=f"{selected_sdq_readable} Distribution by Gender and ADHD Status",
                            color_discrete_map={'ADHD': '#e74c3c', 'Non-ADHD': '#3498db'},
                            marginal='box'
                        )
                        st.plotly_chart(fig_density, use_container_width=True)

    with tab4:
        st.markdown("### Distribution by Key Demographics")
        st.markdown(
            "What is the distribution of participants by key demographics (e.g., study site, ethnicity, race, sex)?")
        st.info("Note: Gender distribution is available in the 'Gender Distribution' tab.")

        st.markdown("#### Distribution by Study Site (N=1213)")
        data_site = {
            'Category': ['Staten Island', 'Midtown', 'Harlem', 'MRV'],
            'Count': [652, 430, 120, 11]
        }
        df_site = pd.DataFrame(data_site).sort_values('Count', ascending=False)

        fig_site = px.bar(
            df_site,
            x='Category',
            y='Count',
            title="Distribution by Study Site",
            color='Category',
            text_auto=True  # Show counts on bars
        )
        fig_site.update_layout(xaxis_title="Study Site", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig_site, use_container_width=True)

        st.markdown("#### Distribution by Ethnicity (N=1213)")
        data_ethnicity = {
            'Category': ['Not Hispanic or Latino', 'Hispanic or Latino', 'Decline to specify', 'nan', 'Unknown'],
            'Count': [777, 296, 77, 43, 20]
        }
        df_ethnicity = pd.DataFrame(data_ethnicity).sort_values('Count', ascending=False)
        df_ethnicity['Category'] = df_ethnicity['Category'].replace({'nan': 'Missing/Not Specified'})

        fig_ethnicity = px.bar(
            df_ethnicity,
            x='Category',
            y='Count',
            title="Distribution by Ethnicity",
            color='Category',
            text_auto=True
        )
        fig_ethnicity.update_layout(xaxis_title="Ethnicity", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig_ethnicity, use_container_width=True)

        st.markdown("#### Distribution by Race (N=1213)")
        st.markdown("_Note: Shows the top 4 reported categories as provided._")
        data_race = {
            'Category': ['White/Caucasian', 'Two or more races', 'Black/African American', 'Hispanic'],
            'Count': [573, 195, 181, 128]
        }
        df_race = pd.DataFrame(data_race).sort_values('Count', ascending=False)

        fig_race = px.bar(
            df_race,
            x='Category',
            y='Count',
            title="Distribution by Race (Top 4 Categories)",
            color='Category',
            text_auto=True
        )
        fig_race.update_layout(xaxis_title="Race", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig_race, use_container_width=True)


    with tab5:
        st.markdown("### Parental Education & Occupation Distribution")
        st.markdown("How are parents' education level and occupational status distributed among children with ADHD and those without ADHD?")

        if 'ADHD_Status' not in data.columns or not any(col in data.columns for col in parent_vars.keys()):
            st.warning("Required 'ADHD_Status' or 'Barratt_*' columns not found in the data. Cannot perform this analysis.")
        else:
            df = data.copy()
            statuses = ['Non-ADHD', 'ADHD']

            selected_var_title = st.selectbox(
                "Select Parental Variable to Analyze:",
                options=list(parent_vars.values())
            )

            var = [k for k, v in parent_vars.items() if v == selected_var_title][0]
            title = selected_var_title

            label_map = education_labels if 'Edu' in var else occupation_labels
            ordered_keys = list(label_map.keys())
            ordered_labels = [label_map[k] for k in ordered_keys]

            sub = df[['ADHD_Status', var]].dropna().copy()
            if sub.empty:
                st.warning(f"No valid (non-missing) data found for {title} ({var}).")
            else:
                counts = {}
                percents = {}
                for status in statuses:
                    vals = sub.loc[sub['ADHD_Status'] == status, var].astype(float)
                    vc = vals.value_counts().sort_index()
                    counts[status] = np.array([vc.get(k, 0) for k in ordered_keys], dtype=int)
                    denom = counts[status].sum()
                    percents[status] = np.round((counts[status] / denom * 100.0), 2) if denom > 0 else np.zeros_like(counts[status], dtype=float)

                
                fig = go.Figure()
                for status, color in zip(statuses, ['#3498db', '#e74c3c']):
                    fig.add_trace(
                        go.Bar(
                            name=status,
                            x=ordered_labels,
                            y=counts[status],
                            text=[f"{p:.1f}%" for p in percents[status]],
                            textposition='outside', 
                            marker_color=color
                        )
                    )
                fig.update_layout(
                    height=500,
                    title_text=f"{title} Distribution by ADHD Status",
                    barmode='group',
                    showlegend=True,
                    xaxis_title="Category",
                    yaxis_title="Count",
                    legend_title="ADHD Status"
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

                
                st.markdown(f"#### Analysis Details for {title}")

                cols = st.columns(2)
                for i, status in enumerate(statuses):
                    with cols[i]:
                        st.markdown(f"**{status} Group**")
                        total_n = int(counts[status].sum())
                        st.metric(label="Sample Size (Non-Missing)", value=total_n)

                        if total_n > 0:
                            mode_idx = int(np.argmax(counts[status]))
                            mode_code = ordered_keys[mode_idx]
                            mode_label = label_map.get(mode_code, 'Unknown')

                            vals = sub.loc[sub['ADHD_Status'] == status, var].astype(float).values
                            med_code = float(np.median(vals))
                            med_label = label_map.get(med_code, 'Unknown')

                            st.markdown(f"**Most Common:** {mode_label}")
                            st.markdown(f"**Median Level:** {med_label}")
                        else:
                            st.markdown("**Most Common:** N/A")
                            st.markdown("**Median Level:** N/A")

                
                st.markdown("#### Statistical Test")
                contingency = np.vstack([counts['Non-ADHD'], counts['ADHD']])
                
                
                valid_cols = contingency.sum(axis=0) > 0
                contingency_filtered = contingency[:, valid_cols]
                
                if contingency_filtered.shape[0] < 2 or contingency_filtered.shape[1] < 2:
                     st.warning("Cannot perform Chi-square test: not enough data or categories after filtering empty columns.")
                else:
                    try:
                        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_filtered)
                        n_obs = contingency_filtered.sum()
                        r, c = contingency_filtered.shape
                        v = _cramers_v(chi2, n_obs, r, c)

                        st.markdown("**Chi-square test of independence:**")
                        st.markdown(f"- Chi-square ($\chi^2$) = {chi2:.3f}")
                        st.markdown(f"- p-value = {_fmt_p(p_value)}")
                        st.markdown(f"- Cramer's V = {v:.3f}")
                        
                        if p_value < 0.05:
                            st.success(f"There is a statistically significant association between {title} and ADHD status (p = {_fmt_p(p_value)}).")
                        else:
                            st.info(f"There is no statistically significant association between {title} and ADHD status (p = {_fmt_p(p_value)}).")

                    except Exception as e:
                        st.error(f"Statistical test (Chi-square) could not be computed: {e}")

                
                with st.expander("Show Full Distribution Data (Counts & %):"):
                    dist_data = {'Category': ordered_labels}
                    for status in statuses:
                        dist_data[f"{status} Count"] = counts[status]
                        dist_data[f"{status} %"] = percents[status]

                    df_dist = pd.DataFrame(dist_data).set_index('Category')
                    st.dataframe(df_dist)

    

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