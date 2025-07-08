import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack
from streamlit_echarts import st_echarts

# Load models
reg = joblib.load("xgb_rating_model.pkl")
clf = joblib.load("xgb_side_effect_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Page Config
st.set_page_config(
    page_title="Drug Predictor Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        body {
            background-color: #0b132b;
            color: #e0e6ed;
        }
        .reportview-container .markdown-text-container {
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #1c2541;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Drug Prediction")
st.markdown("Predict drug effectiveness and side effects with real-time visual feedback.")

# Input form
with st.form("form"):
    col1, col2 = st.columns(2)

    with col1:
        drug_name = st.text_input("Drug Name", "")
        generic_name = st.text_input("Generic Name", "")
        brand_names = st.text_input("Brand Names", "")
        drug_classes = st.text_input("Drug Classes", "")
        related_drugs = st.text_input("Related Drugs", "")

    with col2:
        medical_condition = st.text_input("Medical Condition", "")
        medical_condition_description = st.text_area("Condition Description", "")
        activity = st.number_input("Drug Activity (%)", min_value=0.0, max_value=100.0, value=75.0)
        side_effects = st.text_area("Known Side Effects (optional)", "")

    submitted = st.form_submit_button("🔍 Predict")

# On submit
if submitted:
    # Build input string
    combined_text = " ".join([
        drug_name, generic_name, brand_names, drug_classes,
        related_drugs, side_effects, medical_condition, medical_condition_description
    ])

    X_text_input = tfidf.transform([combined_text])
    X_numeric_input = np.array([[activity]])
    X_input = hstack([X_text_input, X_numeric_input])

    pred_rating = float(reg.predict(X_input)[0])
    pred_side_effect = int(clf.predict(X_input)[0])

    # --- Rating Visualization ---
    st.subheader("📈 Effectiveness Rating")

    rating_options = {
        "tooltip": {"formatter": "{a} <br/>{b} : {c}/10"},
        "series": [{
            "type": "gauge",
            "startAngle": 180,
            "endAngle": 0,
            "min": 0,
            "max": 10,
            "pointer": {"show": True},
            "progress": {"show": True, "width": 15},
            "axisLine": {
                "lineStyle": {
                    "width": 15,
                    "color": [[0.5, "#ef476f"], [0.8, "#ffd166"], [1, "#06d6a0"]]
                }
            },
            "detail": {"formatter": f"{pred_rating:.2f}"},
            "data": [{"value": pred_rating, "name": "Rating"}]
        }]
    }
    st_echarts(rating_options, height="300px")

    # --- Side Effect Prediction ---
    st.subheader("⚠️ Side Effect Prediction")

    pie_options = {
        "tooltip": {"trigger": "item"},
        "legend": {"top": "bottom"},
        "series": [{
            "name": "Risk",
            "type": "pie",
            "radius": ["40%", "70%"],
            "avoidLabelOverlap": False,
            "label": {"show": False},
            "emphasis": {"label": {"show": True, "fontSize": "18"}},
            "labelLine": {"show": False},
            "data": [
                {"value": 1 if pred_side_effect == 1 else 0, "name": "Side Effects", "itemStyle": {"color": "#ff6b6b"}},
                {"value": 1 if pred_side_effect == 0 else 0, "name": "No Side Effects", "itemStyle": {"color": "#4ecdc4"}}
            ]
        }]
    }
    st_echarts(pie_options, height="300px")

    # --- Summary ---
    st.markdown("---")
    st.success(f"✅ **Predicted Rating:** {pred_rating:.2f} / 10")
    st.info("📊 High rating indicates good effectiveness.")

    if pred_side_effect:
        st.error("⚠️ Likely to cause side effects.")
    else:
        st.success("✅ Unlikely to cause side effects.")

    st.caption("This dashboard uses a machine learning model trained on real drug data from Drugs.com.(Data might be inaccurate, this is just for learning purposes)")

