import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack

# Load models
reg = joblib.load('xgb_rating_model.pkl')
clf = joblib.load('xgb_side_effect_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# --- Page Configuration ---
st.set_page_config(
    page_title="Drug Analysis",
    page_icon="💉",
    layout="centered"
)

# --- Custom Styling ---
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            background-color: #0f1c2e;
            color: #d8e3e7;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 800px;
            margin: auto;
        }

        .stTextInput input, .stTextArea textarea, .stNumberInput input {
            background-color: #1e2a38;
            color: white;
            border-radius: 0.5rem;
            border: 1px solid #3a506b;
            padding: 0.5rem;
        }

        .stButton>button {
            background-color: #3a86ff;
            color: white;
            border-radius: 0.5rem;
            padding: 0.6rem 1rem;
            font-weight: 600;
            transition: 0.3s ease;
            width: 100%;
        }

        .stButton>button:hover {
            background-color: #00b4d8;
            color: black;
        }

        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #caffbf;
        }

        .stCaption, .stSubheader, .stMarkdown p {
            color: #adb5bd;
        }

        .result-box {
            background-color: #172a3a;
            padding: 1rem;
            border-radius: 0.7rem;
            border-left: 4px solid #00b4d8;
            margin-top: 1rem;
        }

        .stAlert {
            border-radius: 0.6rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("Drug Effectiveness & Side Effect Predictor")
st.markdown("Use this AI-powered tool to estimate a drug's effectiveness and potential side effects based on its description and activity.")

# --- Input Form ---
with st.form("prediction_form"):
    st.subheader("🧾 Enter Drug Information")

    col1, col2 = st.columns(2)
    with col1:
        drug_name = st.text_input("Drug Name", help="Name of the drug (e.g., Ibuprofen)")
        generic_name = st.text_input("Generic Name", help="Active compound (e.g., Ibuprofen)")
        brand_names = st.text_input("Brand Names", help="Comma-separated brand names")
        related_drugs = st.text_input("Related Drugs", help="List of similar drugs (if any)")

    with col2:
        drug_classes = st.text_input("Drug Classes", help="E.g., NSAID, Antihistamine")
        medical_condition = st.text_input("Medical Condition", help="Treated condition (e.g., Migraine)")
        activity = st.number_input("Activity (%)", min_value=0.0, max_value=100.0, value=75.0,
                                   help="Estimated pharmacological activity of the drug (0-100%)")

    medical_condition_description = st.text_area("Medical Condition Description",
                                                 help="Short description of the medical condition.")
    side_effects = st.text_area("Known Side Effects (optional)",
                                help="Mention any known side effects.")

    submitted = st.form_submit_button("🔍 Predict")

# --- Prediction & Output ---
if submitted:
    with st.spinner("Running predictions..."):
        combined_text = " ".join([
            drug_name, generic_name, brand_names, drug_classes,
            related_drugs, side_effects, medical_condition, medical_condition_description
        ])

        X_text_input = tfidf.transform([combined_text])
        X_numeric_input = np.array([[activity]])
        X_input = hstack([X_text_input, X_numeric_input])

        pred_rating = reg.predict(X_input)[0]
        pred_side = clf.predict(X_input)[0]

    st.subheader("📊 Prediction Results")

    with st.container():
        st.markdown(f"""
        <div class="result-box">
            <h3>⭐ Predicted Effectiveness Rating: <span style="color:#ffd60a;">{pred_rating:.2f} / 10</span></h3>
            {"<p style='color:#95d5b2;'>🌟 Highly effective</p>" if pred_rating >= 8 else 
             "<p style='color:#f4a261;'>⚠️ Moderate effectiveness</p>" if pred_rating >= 5 else
             "<p style='color:#f94144;'>❌ Low effectiveness</p>"}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-box">
            <h3>💥 Predicted Side Effects Risk: 
                <span style="color:{'#f94144' if pred_side else '#80ed99'};">{'Yes' if pred_side else 'No'}</span>
            </h3>
            {"<p style='color:#f94144;'>🚨 Potential for side effects.</p>" if pred_side else "<p style='color:#80ed99;'>✅ No major side effects predicted.</p>"}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("🔍 These predictions are AI-generated using clinical text + activity data. For informational use only, not a substitute for medical advice.")
