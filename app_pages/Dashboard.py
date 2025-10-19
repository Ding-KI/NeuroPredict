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

def main():
    st.title("NeuroPredict Dashboard")
    
    # Dataset Description Section
    st.header("Dataset")
    st.markdown("""
This project uses a publicly available dataset originally designed to explore associations between ADHD symptoms and various behavioral, cognitive, and demographic factors. The data were collected through several standardized psychological instruments and questionnaires administered to children and adolescents.

Three main questionnaires involved:
1. **EHQ (Edinburgh Handedness Questionnaire)** – a standardized tool used to measure hand preference for different activities such as writing, drawing, throwing, and using scissors. Scores range from -100 (completely left-handed) to +100 (completely right-handed), representing a Laterality Index.
2. **APQ (Alabama Parenting Questionnaire)** – assesses parenting practices, including positive involvement, supervision, discipline, and corporal punishment. Responses are rated on a Likert scale from 1 ("Never") to 5 ("Always").
3. **SDQ (Strengths and Difficulties Questionnaire)** – measures behavioral and emotional problems in children, including hyperactivity, emotional symptoms, conduct problems, peer relationships, and prosocial behavior. Items are rated as 0 ("Not True"), 1 ("Somewhat True"), or 2 ("Certainly True").

The raw data contains three sub-datasets:
- The **QUANTITATIVE METADATA** dataset contains questionnaire scores, test results, and MRI scan information.
- The **CATEGORICAL METADATA** dataset includes demographic variables such as ethnicity, race, and parental education.
- The **SOLUTIONS dataset records** participants' sex and ADHD diagnostic outcomes.

All three datasets share the same **Participant ID** sequence, allowing us to merge them by ID in the following step.
    """)
    
    # Literature Section
    st.header("Literature")
    st.markdown("""
Quinn PO, Madhoo M. A Review of Attention-Deficit/Hyperactivity Disorder in Women and Girls: Uncovering This Hidden Diagnosis. Primary care companion for CNS disorders. 2014;16(3)

Ayano G, Demelash S, Gizachew Y, Tsegay L, Alati R. The global prevalence of attention deficit hyperactivity disorder in children and adolescents: An umbrella review of meta-analyses. Journal of affective disorders. 2023;339:860–6.

Why ADHD Is Often Underdiagnosed In Women[Blog on the Internet]. Michigan: Henry Ford Health Staff; 2023 Sep 7-. [Cited 2025 Sep 15]. Available from: https://www.henryford.com/blog/2023/09/why-adhd-is-often-underdiagnosed-in-women

Barkley RA. Global Issues Related to the Impact of Untreated Attention-Deficit/Hyperactivity Disorder From Childhood to Young Adulthood. Postgraduate medicine. 2008;120(3):48–59.

Karolinska Institutet. Strategy 2030 – creating Karolinska Institutet's future together [Internet]. Stockholm: The Karolinska Institutet University Board; 2019[cited 2025 Sep 21].Availablefrom:https://staff.ki.se/our-ki/strategy-2030-creating-karolinska-institutets-future-together

Folkhälsomyndigheten. A framework for implementing and monitoring the National Public Health Policy [Internet]. Sweden: Public Health Agency of Sweden; 2021 [cited 2025 Sep 21]. Availablefrom:https://www.folkhalsomyndigheten.se/contentassets/bb50d995b033431f9574d61992280e61/towards-good-equitable-health.pdf

Swedish Work Environment Authority. Swedish labour market model and collective agreements. Sweden; [updated 2025 July 2; cited 2025 Sep 21]. Available from: https://www.av.se/en/work-environment-work-and-inspections/foreign-labour-in-sweden/posting-foreign-labour-in-sweden/swedish-labour-market-model-and-collective-agreements/

Government Offices of Sweden. Working Hours Act (Arbetstidslagen) [Internet]. Sweden: Ministry of Employment; 2015 [cited 2025 Sep 21]. Available from: https://www.government.se/government-policy/labour-law-and-work-environment/1982673-working-hours-act-arbetstidslagen/

Widsdatathon. "WiDS Datathon 2025: Unraveling the Mysteries of the Female Brain: Sex Patternsin ADHD." Kaggle, 2025. Available from: https://www.kaggle.com/competitions/widsdatathon2025/data.

A roadmap to implementing machine learning in healthcare: from concept to practice. https://pubmed-ncbi-nlm-nih-gov.proxy.kib.ki.se/39906065/

Machine Learning Operations in Health Care: A Scoping Review https://pubmed-ncbi-nlm-nih-gov.proxy.kib.ki.se/40206123/

Artificial intelligence in healthcare: transforming the practice of medicine. Available from: https://pmc-ncbi-nlm-nih-gov.proxy.kib.ki.se/articles/PMC8285156/

Algorithm fairness in artificial intelligence for medicine and healthcare. Available from: https://pubmed-ncbi-nlm-nih-gov.proxy.kib.ki.se/37380750/

Beyond probability-impact matrices in project risk management: A quantitative methodology for risk prioritisation. Available from: https://research.ebsco.com/c/qzil4s/search/details/h2oq75yghz?limiters=FT%3AY&q=Beyond%20probability-impact%20matrices%20in%20project%20risk%20management%3A%20A%20quantitative%20methodology%20for%20risk%20prioritisation&searchMode=all

A Guide to the Project Management Body of Knowledge (PMBOK® Guide) – Seventh Edition and The Standard for Project Management (ENGLISH). Available from: https://research-ebsco-com.ezp.sub.su.se/c/qzil4s/search/details/ivlj2dlhyv?limiters=FT%3AY&q=PMBOK%C2%AE+Guide+%28Project+Management+Institute%29&searchMode=all

Assuring the Machine Learning Lifecycle : Desiderata, Methods, and Challenges. Available from:https://research-ebsco-com.ezp.sub.su.se/c/qzil4s/search/details/bomzqh2cfr?limiters=FT%3AY&q=Assuring%20the%20Machine%20Learning%20Lifecycle%3A%20Desiderata%2C%20Methods%2C%20and%20Challenges&searchMode=all
    """)
    
    # Group Members Section
    st.header("Group Members")
    st.markdown("""
- Laura Lemetti
- Md Imran Mansur
- Kaviya Palaniyappan
- Ding Xiao
- Songyue Xie
    """)
    
    # Contact Section
    st.header("Contact Information")
    st.markdown("""
- apply.ding.xiao@gmail.com
- xmxsy123@outlook.com
    """)

if __name__ == "__main__":
    main()
