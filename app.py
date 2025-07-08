import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack

# Load models
reg = joblib.load('xgb_rating_model.pkl')
clf = joblib.load('xgb_side_effect_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# --- Page Config ---
st.set_page_config(
    page_title="Drug Effectiveness & Side Effect Predictor",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Dark Theme Styling ---
st.markdown("""
    <style>
        body {
            background-color: #0b132b;
            color: #e0e6ed;
        }
        .stTextInput, .stTextArea, .stNumberInput {
            background-color: #1c2541;
            color: white;
        }
        .main {
            background-color: #0b132b;
            padding: 2rem;
        }
        .stButton>button {
            background-color: #3a506b;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #5bc0be;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("💊 Drug Effectiveness & Side Effect Predictor")
st.markdown("Predict how effective a drug might be and whether it may cause side effects based on its properties.")

# --- Input Form ---
with st.form("prediction_form"):
    st.subheader("🔬 Drug Information")

    drug_name = st.text_input("Drug Name", help="Enter the name of the drug (e.g., Aspirin)")
    generic_name = st.text_input("Generic Name", help="Main active ingredient or compound")
    brand_names = st.text_input("Brand Names", help="List common brand names, separated by commas")
    drug_classes = st.text_input("Drug Classes", help="Therapeutic or pharmacologic class (e.g., NSAID, antibiotic)")
    related_drugs = st.text_input("Related Drugs", help="Any known related or similar drugs")
    medical_condition = st.text_input("Medical Condition", help="Condition this drug is used to treat (e.g., Headache)")
    medical_condition_description = st.text_area("Medical Condition Description", help="Brief description of the condition")
    activity = st.number_input("Activity (%)", min_value=0.0, max_value=100.0, value=80.0, help="Estimated pharmacological activity of the drug in %")
    side_effects = st.text_area("Known Side Effects (optional)", help="Known or documented side effects (if any)")

    submit = st.form_submit_button("🔍 Predict")

# --- On Submit ---
if submit:
    # Combine all inputs into one text string
    combined_text = " ".join([
        drug_name, generic_name, brand_names, drug_classes,
        related_drugs, side_effects, medical_condition, medical_condition_description
    ])

    # Vectorize and format inputs
    X_text_input = tfidf.transform([combined_text])
    X_numeric_input = np.array([[activity]])
    X_input = hstack([X_text_input, X_numeric_input])

    # Predict
    pred_rating = reg.predict(X_input)[0]
    pred_side = clf.predict(X_input)[0]

    st.subheader("📊 Results")
    
    # Display predictions with interpretation
    st.markdown(f"**Predicted Effectiveness Rating:** `{pred_rating:.2f}` out of 10")
    if pred_rating >= 8:
        st.success("🌟 The drug is predicted to be highly effective.")
    elif pred_rating >= 5:
        st.warning("⚠️ The drug may be moderately effective.")
    else:
        st.error("❌ The drug is predicted to have low effectiveness.")

    st.markdown(f"**Side Effect Risk:** `{ 'Yes' if pred_side == 1 else 'No' }`")
    if pred_side == 1:
        st.warning("🚨 This drug is likely to have side effects. Monitor carefully.")
    else:
        st.success("✅ No significant side effects predicted.")

    st.markdown("---")
    st.caption("💡 This model uses both textual and numeric data for prediction. Results are for informational purposes only and not a substitute for medical advice.")

