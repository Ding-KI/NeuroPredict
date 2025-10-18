# NeuroPredict Dashboard

![NeuroPredict Logo](./Neu.png)

**An Interactive Machine Learning Dashboard for ADHD Diagnosis Prediction**

NeuroPredict is a comprehensive web-based dashboard designed to analyze and predict ADHD (Attention-Deficit/Hyperactivity Disorder) diagnosis using machine learning techniques. The project leverages behavioral, parenting, and demographic data collected through standardized psychological instruments to provide insights into ADHD patterns and develop predictive models.

## Introduction

**The Problem:**
ADHD is one of the most common neurodevelopmental disorders in children and adolescents, affecting approximately 5-7% of the global population. However, ADHD diagnosis is often underdiagnosed, particularly in females, due to different symptom presentations and societal biases. Early and accurate diagnosis is crucial for effective intervention and treatment.

**The Solution:**
NeuroPredict addresses this challenge by providing a data-driven approach to ADHD analysis and prediction. The dashboard integrates multiple data sources including:
- **EHQ (Edinburgh Handedness Questionnaire)** - Hand preference measurements
- **APQ (Alabama Parenting Questionnaire)** - Parenting practices assessment  
- **SDQ (Strengths and Difficulties Questionnaire)** - Behavioral and emotional problem evaluation

The solution is valuable because it provides healthcare professionals and researchers with an interactive tool to explore ADHD patterns, understand contributing factors, and make data-informed decisions about diagnosis and treatment planning.

## System description

### Project Architecture
```text
NeuroPredict/
├── app.py                        # Main Streamlit application entry point
├── app_pages/                    # Streamlit page modules
│   ├── Dashboard.py              # Main dashboard page
│   ├── Descriptive.py            # Descriptive analysis (5+ research questions)
│   ├── Diagnostic.py             # Diagnostic analysis (3+ research questions)
│   ├── Predictive.py             # Predictive modeling and analysis
│   └── Prescriptive.py           # SHAP explanations and model interpretability
├── data/                         # Data storage
│   ├── raw_data/                 # Original datasets
│   │   └── TRAIN/                # Training data files
│   │       ├── TRAIN_CATEGORICAL_METADATA_new.xlsx
│   │       ├── TRAIN_QUANTITATIVE_METADATA_new.xlsx
│   │       └── TRAINING_SOLUTIONS.xlsx
│   └── processed_data/           # Preprocessed data
│       └── df_preprocessed.csv   # Cleaned and standardized dataset
├── jupyter-notebooks/            # Data analysis and modeling notebooks
│   ├── EDA & Modelling.ipynb    # Exploratory data analysis and model development
│   └── Figs/                     # Generated analysis figures
│       ├── Question3_*.html      # APQ distribution analysis
│       ├── Question4_*.html      # SDQ distribution analysis
│       ├── Question5_*.html      # Parent background analysis
│       ├── Question6_*.html      # Feature correlation analysis
│       ├── Question7_*.html      # APQ correlation analysis
│       └── Question8_*.html      # SDQ-APQ correlation analysis
├── model/                        # Trained machine learning models
│   └── best_model_Decision_Tree_Depth=3.joblib
├── Neu.png                       # Project logo
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

### Dependencies

**Core Requirements:**
- Python 3.8+ (Tested on Python 3.9+)
- Streamlit 1.46.1 - Web application framework
- Scikit-learn 1.7.0 - Machine learning algorithms
- SHAP 0.48.0 - Model interpretability and explanations
- Plotly Express 0.4.1 - Interactive data visualization
- Seaborn 0.13.2 - Statistical data visualization
- Jupyter 1.1.1 - Interactive notebook environment
- Watchdog 6.0.0 - File system monitoring

**Data Processing:**
- Pandas - Data manipulation and analysis
- NumPy - Numerical computing
- Scipy - Scientific computing and statistics

**Visualization:**
- Matplotlib - Static plotting
- Plotly - Interactive plotting

### Installation

Follow these steps to set up the NeuroPredict dashboard on your local machine:

**Prerequisites:**
- Python 3.8 or higher
- Git (for cloning the repository)
- For Mac users: Xcode Command Line Tools may be required. See [Streamlit documentation](https://docs.streamlit.io/get-started/installation/command-line#prerequisites) for details.

**Setup Instructions:**

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd NeuroPredict
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv neuropredict_env
   ```

3. **Activate the virtual environment:**
   - **Windows (Command Prompt):** `neuropredict_env\Scripts\activate.bat`
   - **Windows (PowerShell):** `neuropredict_env\Scripts\Activate.ps1`
   - **Linux/Mac:** `source neuropredict_env/bin/activate`

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installation:**
   ```bash
   streamlit hello
   ```

6. **Run the NeuroPredict dashboard:**
   ```bash
   streamlit run app.py
   ```

### Execution

**Quick Start:**
```bash
streamlit run app.py
```

**Alternative execution methods:**
```bash
# If the above command fails, try:
python -m streamlit run app.py

# Or run the main dashboard directly:
streamlit run app_pages/Dashboard.py
```

The dashboard will be available at `http://localhost:8501` in your web browser.

### Dashboard Features

**📊 Descriptive Analysis:**
- APQ questionnaire score distributions across genders and ADHD groups
- SDQ behavioral patterns analysis
- Parent background characteristics (education, occupation)
- Comprehensive statistical summaries with effect sizes

**🔍 Diagnostic Analysis:**
- Feature correlation analysis with ADHD diagnosis
- Behavioral and demographic factor associations
- Statistical significance testing and confidence intervals

**🎯 Predictive Analysis:**
- Machine learning model training and evaluation
- ADHD diagnosis prediction using Decision Tree algorithms
- Model performance metrics and validation

**💡 Prescriptive Analysis:**
- SHAP (SHapley Additive exPlanations) model interpretability
- Individual prediction explanations
- Feature importance analysis across different demographic groups
- Interactive force plots and waterfall charts

### Data Pipeline

⚠️ **Important:** The dashboard requires pre-trained models to function properly. Before using the predictive and prescriptive features:

1. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook jupyter-notebooks/EDA\ \&\ Modelling.ipynb
   ```

2. **Execute all cells** to generate:
   - Preprocessed dataset (`data/processed_data/df_preprocessed.csv`)
   - Trained machine learning models (`model/best_model_*.joblib`)
   - Analysis figures (`jupyter-notebooks/Figs/`)

This follows a standard data science pipeline where notebooks handle data exploration and model development, while the Streamlit dashboard provides the user interface for model deployment and interpretation. 

## Contributors

Laura Lemetti, Ding Xiao, Md Imran Mansur, Kaviya Palaniyappan, Songyue Xie
