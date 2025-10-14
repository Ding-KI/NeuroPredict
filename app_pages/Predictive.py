import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

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
def load_model(model_path):
    """Loads the pre-trained model"""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Define the questions and their corresponding options and values
gender_question = {"16. Gender": "Please select the gender."}
gender_options = {'Male': 0, 'Female': 1}

apq_questions = {
    "1. You slap your child when he/she has done something wrong.": "(Parenting Corporal Punishment Score)",
    "2. The punishment you give your child depends on your mood.": "(Parenting Inconsistent Discipline Score)",
    "3. You regularly attend PTA meetings, parent/teacher conferences, or other meetings at your child's school.": "(Parenting Involvement Score)",
    "4. You take away privileges or money from your child as punishment.": "(Parenting Other Discipline Practices Score)",
    "5. Your child comes home from school more than an hour past the time you expect him/her to be home.": "(Parenting Poor Monitoring/Supervision Score)",
    "6. You compliment your child when he/she has done something well.": "(Parenting Positive Parenting Score)",
}
apq_options = {'Never': 0, 'Almost never': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4}

sdq_questions = {
    "7. Often has temper tantrums or hot tempers.": "(Conduct Problems Scale)",
    "8. Generally well behaved, usually does what adults request.": "(Total Difficulties Scale)",
    "9. Many worries, or often seems worried.": "(Emotional Problems Scale)",
    "10. Often fights with other children or bullies them.": "(Externalizing Scale)",
    "11. Restless, overactive, cannot stay still for long.": "(Hyperactivity Scale)",
    "12. Often unhappy, depressed or tearful.": "(Internalizing Scale)",
    "13. Picked on or bullied by other children.": "(Peer Problems Scale)",
    "14. Often offers to help others (parents, teachers, children).": "(Prosocial Scale)",
    "15. Do these difficulties upset or distress your child?": "(Generating Impact Scale)",
}
sdq_options = {'Not True': 0, 'Somewhat True': 1, 'Certainly True': 2}

gender_question = {"16. Gender": "Please select the gender."}
gender_options = {'Male': 0, 'Female': 1}

def main():
    """
    This function defines the Streamlit page for the ADHD prediction dashboard.
    """
    st.title("ADHD Outcome Prediction Questionnaire")
    st.markdown("""
    Please answer the following questions based on your observations. Your responses will be used to predict an outcome using a machine learning model.
    **Disclaimer:** This is a tool for preliminary assessment and is **not** a substitute for a professional medical diagnosis.
    """)

    # Load the model
    model = load_model('/Users/kaviya/Repositories/ProHi/NeuroPredict copy/model/best_model_Decision_Tree_Depth=3.joblib')

    # Use a form to collect all inputs and submit them at once
    with st.form("prediction_form"):
            st.header("Parenting Questionnaire (APQ)")
            apq_responses = {}
            for key, question in apq_questions.items():
                full_label = f"**{key}**\n\n{question}"
                response = st.radio(full_label, options=list(apq_options.keys()), horizontal=True, key=key)
                apq_responses[key] = response
            st.markdown("---")

            st.header("Strengths and Difficulties Questionnaire (SDQ)")
            sdq_responses = {}
            for key, question in sdq_questions.items():
                full_label = f"**{key}**\n\n{question}"
                response = st.radio(full_label, options=list(sdq_options.keys()), horizontal=True, key=key)
                sdq_responses[key] = response

            st.header("Gender")
            gender_response = st.radio(
                   "**1. Please select the gender:**",
                   options=list(gender_options.keys()),
                   horizontal=True,
                   key="gender") 
            submitted = st.form_submit_button("Submit and Predict Outcome")
        
    if submitted:
        if model is None:
             st.warning("Cannot proceed with prediction because the model is not loaded.")
        else:
            input_data = []
            
            # 1. APQ responses
            for key in apq_questions.keys():
                input_data.append(apq_options[apq_responses[key]])

            # 2. SDQ responses
            for key in sdq_questions.keys():
                input_data.append(sdq_options[sdq_responses[key]])

            # 3. Gender response
            input_data.append(gender_options[gender_response])
            
            # Convert to a NumPy array and reshape for the model
            final_features = np.array(input_data).reshape(1, -1)
            
            # Make prediction
            try:
                prediction = model.predict(final_features)
                prediction_proba = model.predict_proba(final_features)

                st.subheader("Prediction Result")
                
                # Assuming the model outputs 1 for 'High Likelihood' and 0 for 'Low Likelihood'
                if prediction[0] == 1:
                    st.error("Prediction: High Likelihood of ADHD")
                else:
                    st.success("Prediction: Low Likelihood of ADHD")
                
                st.write("Confidence Score:")
                st.info(f"The model is {prediction_proba[0][prediction[0]]*100:.2f}% confident in this prediction.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
    
    st.markdown("---")
    st.markdown(
       """
       <div style='text-align: center; color: #666; font-size: 0.9rem; padding: 2rem;'>
           <h4>NeuroPredict Dashboard</h4>
           <p>Built by Group 4 | Last Updated: {}</p>
       </div>
       """.format(datetime.now().strftime("%Y-%m-%d %H:%M")),
       unsafe_allow_html=True)

if __name__ == "__main__":
    main()


    
