# PROHI Dashboard Example

# [NeuroPredict]

_You can modify this README file with all the information that your team consider relevant for a technical audience who would like to understand your project or to run it in the future._

_Note that this file is written in **MarkDown** language. A reference is available here: <https://www.markdownguide.org/basic-syntax/>_

Include the name, logo and images refering to your project

![Your dashboard](./assets/example-image.jpg)

## Introduction

NeuroPredict is an interactive web dashboard to.... 

The problem detected was...

The proposed solution is valuable because...

## System description

### Structure 
NeuroPredict/
├── Dashboard.py                    # main page
├── pages/                         # Streamlit page
│   ├── 1_📊_Descriptive.py        # descriptive analysis (min. 5 questions)
│   ├── 2_🔍_Diagnostic.py         # diagnostic analysis（min. 3 questions）
│   ├── 3_🎯_Predictive.py         # predictive analysis
│   ├── 4_💡_Prescriptive.py       # SHAP explaintion
│   └── 5_ℹ️_About.py              # About page
├── src/                           # core func module
│   ├── data/                      # data processing
│   │   ├── loader.py             # data loader
│   │   └── preprocessor.py       # data preprocess
│   ├── analysis/                  # analysis
│   │   ├── descriptive.py        # descriptive func
│   │   ├── diagnostic.py         # diagnostic func
│   │   └── statistical.py        # statisc func
│   ├── models/                    # machine learning module
│   │   ├── model_trainer.py      # model training
│   │   ├── predictor.py          # predictor interface
│   │   └── evaluator.py          # model evaluation
│   └── visualization/            # visualization
│       ├── plotly_charts.py      # Plotly
│       └── shap_plots.py         # SHAP
├── data/                         # data
│   ├── raw/                      # raw data
│   └── processed/                # process data
├── models/                       # saved machine learning model
│   └── trained_models/          # trained model（pickle）
├── notebooks/                    # Jupyter
│   ├── EDA & Modelling.ipynb    # Jupyter notebook
├── tests/                        # test doc
├── requirements.txt              # dependencies
└── README.md                     # docmentation

### Dependencies

Tested on Python < 2.x
to be added soon...

### Installation

Run the commands below in a terminal to configure the project and install the package dependencies for the first time.

If you are using Mac, you may need to follow install Xcode. Check the official Streamlit documentation [here](https://docs.streamlit.io/get-started/installation/command-line#prerequisites). 

1. Create the environment with `python -m venv env`
2. Activate the virtual environment for Python
   - `source env/bin/activate` [in Linux/Mac]
   - `.\env\Scripts\activate.bat` [in Windows command prompt]
   - `.\env\Scripts\Activate.ps1` [in Windows PowerShell]
3. Make sure that your terminal is in the environment (`env`) not in the global Python installation
4. Install required packages `pip install -r ./requirements.txt`
5. Check that everything is ok running `streamlit hello`

### Execution

To run the dashboard execute the following command:

```
> streamlit run Dashboard.py
# If the command above fails, use:
> python -m streamlit run Dashboard.py
```


### Creating pre-trained models for the web dashboadr 

⚠️ **NOTE:** In the predictive analytics tab, the web dashboard is looking for a pre-trained model in the folder `assets/`. The first time that you execute the application, it will show an error saying that such file does not exist. Therefore, you need to execute the notebook inside the folder `jupyter-notebook/` to create the pre-trained model.

This logic resembles the expected pipeline, where the jupyter notebooks are used to iterate the data modeling part until a satisfactory trained model is created, and the streamlit scripts are only in charge of rendering the user-facing interface to generate the prediction for new data. In practice, the data science pipeline is completely independent from the web dashboard, and both are connected via the pre-trained model. 

## Contributors

_Add the project's authors, contact information, and links to their websites or portfolios._
Laura Lemetti, Ding Xiao, Md Imran Mansur, Kaviya Palaniyappan, Songyue Xie
