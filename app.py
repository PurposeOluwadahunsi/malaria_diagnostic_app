# app.py
"""
Final corrected Streamlit app for Malaria Risk Detector
- Ensures prediction state is saved in st.session_state so feedback insert runs after rerun
- Yes/No inputs (converted to 1/0)
- Smooth "analyzing" progress animation
- Battery-style probability display (visual + numeric)
- Feedback form submit inside st.form that reads prediction from session_state
- Supabase insertion via REST API with robust logging (terminal + UI debug panel)
- Removed feature-importance display
- IMPORTANT: This expects your Supabase table `malaria_feedback` with columns:
    probability (integer), prediction (text), helpful (text), clinic_result (text), comment (text)
- Place `Malaria_Diagnostic_Model.pkl` in the same folder as this file.
- Prefer setting SUPABASE_URL and SUPABASE_ANON_KEY as environment variables for safety.
"""

import os
import time
import json
import pickle
import traceback
from typing import Tuple, Dict, Any

import requests
import numpy as np
import streamlit as st

# ----------------------
# Configuration (env preferred)
# ----------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://cziobwtgiqsyupkjuyfv.supabase.co")
SUPABASE_ANON_KEY = os.getenv(
    "SUPABASE_ANON_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN6aW9id3RnaXFzeXVwa2p1eWZ2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njk5NDQyMzcsImV4cCI6MjA4NTUyMDIzN30._u-sC1IKGQlzbgTUuWQ-ldHjy8crD5U7hc6gYgcqGEs",
)
SUPABASE_ENDPOINT = f"{SUPABASE_URL}/rest/v1/malaria_feedback"

MODEL_PATH = "Malaria_Diagnostic_Model.pkl"

# ----------------------
# Streamlit page + CSS
# ----------------------
st.set_page_config(page_title="Malaria Risk Detector", page_icon="🦟", layout="wide")

st.markdown(
    """
    <style>
      .header { font-size:34px; font-weight:700; color:#0b6e4f; margin-bottom:6px; }
      .subheader { font-size:13px; color:#333; margin-top:-6px; margin-bottom:12px; }
      .card { background:#fff; padding:16px; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.06); }
      .risk-badge { font-weight:700; padding:6px 12px; border-radius:20px; color:white; display:inline-block; }
      .battery { width:70%; background:#e6e6e6; border-radius:8px; height:36px; position:relative; overflow:hidden; }
      .battery-inner { height:100%; border-radius:8px; text-align:center; color:white; font-weight:700; line-height:36px; transition: width 0.5s ease; }
      .battery-tip { position:absolute; right:-10px; top:8px; width:4px; height:20px; background:#999; border-radius:2px; }
      .debug-box { background:#f7f7f7; padding:10px; border-radius:6px; font-family:monospace; white-space:pre-wrap; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="header"> Malaria Risk Detector</div>', unsafe_allow_html=True)


# ----------------------
# Load model
# ----------------------
@st.cache_resource
def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        return None, f"Model file not found at {path}"
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {e}"

model, model_err = load_model()

# ----------------------
# Feature order (MUST match training)
# ----------------------
FEATURE_ORDER = [
    "Age",
    "Fever",
    "Headache",
    "Abdominal_Pain",
    "General_Body_Malaise",
    "Dizziness",
    "Vomiting",
    "Confusion",
    "Backache",
    "Chest_pain",
    "Coughing",
    "Joint_Pain",
    "Sex_Male",
]

# ----------------------
# Helpers
# ----------------------
def yesno_to_int(choice: str) -> int:
    return 1 if str(choice).lower() in ("yes", "true", "1") else 0

def prob_to_risk(prob: float) -> Tuple[str, str]:
    if prob < 0.30:
        return "Low Risk", "#2ecc71"
    elif prob < 0.61:
        return "Medium Risk", "#f1c40f"
    else:
        return "High Risk", "#e74c3c"

def supabase_insert(record: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    try:
        resp = requests.post(SUPABASE_ENDPOINT, headers=headers, json=[record], timeout=12)
        # Terminal logs for debugging
        print("Supabase POST ->", SUPABASE_ENDPOINT)
        print("Request headers:", {k: v for k, v in headers.items() if k != "apikey"})
        print("Payload:", json.dumps([record], ensure_ascii=False))
        print("Response status:", resp.status_code)
        print("Response text:", resp.text)
        try:
            resp_json = resp.json()
        except Exception:
            resp_json = {"text": resp.text}
        if resp.status_code in (200, 201):
            return True, resp_json
        else:
            return False, {"status_code": resp.status_code, "response": resp_json}
    except Exception as e:
        print("Supabase insert exception:", str(e))
        return False, {"error": str(e), "trace": traceback.format_exc()}

# ----------------------
# Initialize session state containers
# ----------------------
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None
if "last_probability" not in st.session_state:
    st.session_state["last_probability"] = None
if "last_prob_pct" not in st.session_state:
    st.session_state["last_prob_pct"] = None
if "last_features" not in st.session_state:
    st.session_state["last_features"] = None
if "debug_logs" not in st.session_state:
    st.session_state["debug_logs"] = []
if "feedback_list" not in st.session_state:
    st.session_state["feedback_list"] = []

def log_debug(msg: str):
    print(msg)  # terminal
    st.session_state["debug_logs"].append(msg)

# ----------------------
# UI layout
# ----------------------
left_col, right_col = st.columns([1.2, 1.0])

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Patient information")
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=25, step=1, format="%d")
    sex = st.radio("Sex", ["Male", "Female"], index=0, horizontal=True)

    st.markdown("---")
    st.subheader("Answer symptoms (Yes / No)")

    fever = st.radio("Do you have Fever?", ["No", "Yes"], index=0, horizontal=True)
    headache = st.radio("Do you have Headache?", ["No", "Yes"], index=0, horizontal=True)
    abdominal_pain = st.radio("Do you have Abdominal Pain?", ["No", "Yes"], index=0, horizontal=True)
    general_body_malaise = st.radio("Do you have General Body Malaise?", ["No", "Yes"], index=0, horizontal=True)
    dizziness = st.radio("Do you feel dizzy sometimes?", ["No", "Yes"], index=0, horizontal=True)
    vomiting = st.radio("Are you vomitting?", ["No", "Yes"], index=0, horizontal=True)
    confusion = st.radio("Are you confused about any symptoms?", ["No", "Yes"], index=0, horizontal=True)
    backache = st.radio("Do you have Backache?", ["No", "Yes"], index=0, horizontal=True)
    chest_pain = st.radio("Do you have Chest Pain?", ["No", "Yes"], index=0, horizontal=True)
    coughing = st.radio("Do you cough?", ["No", "Yes"], index=0, horizontal=True)
    joint_pain = st.radio("Do you perharps have Joint Pain?", ["No", "Yes"], index=0, horizontal=True)

    st.markdown("---")
    st.info("Tip: please answer honestly, this tool is only a probability-based risk estimator, not a diagnosis.")
    analyze_btn = st.button("Analyze Symptoms", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Analysis result")
    result_area = st.empty()
    battery_area = st.empty()
    advice_area = st.empty()
    st.markdown("---")
    st.subheader("Feedback")
    feedback_area = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# When Analyze is clicked -> run model and store results in session_state
# ----------------------
if analyze_btn:
    if model_err:
        st.error(f"Model load error: {model_err}")
        log_debug(f"Model load error: {model_err}")
    elif model is None:
        st.error("Model not found. Ensure 'Malaria_Diagnostic_Model.pkl' is in this folder.")
        log_debug("Model not found.")
    else:
        sex_male = 1 if sex == "Male" else 0
        features = [
            int(age),
            yesno_to_int(fever),
            yesno_to_int(headache),
            yesno_to_int(abdominal_pain),
            yesno_to_int(general_body_malaise),
            yesno_to_int(dizziness),
            yesno_to_int(vomiting),
            yesno_to_int(confusion),
            yesno_to_int(backache),
            yesno_to_int(chest_pain),
            yesno_to_int(coughing),
            yesno_to_int(joint_pain),
            sex_male,
        ]

        # Save feature vector to session so feedback can access it after rerun
        st.session_state["last_features"] = dict(zip(FEATURE_ORDER, features))
        log_debug("Feature vector: " + json.dumps(st.session_state["last_features"], ensure_ascii=False))

        # progress animation
        prog = st.progress(0)
        status = st.empty()
        status.info("Analyzing symptoms...")
        for p in range(0, 101, 4):
            time.sleep(0.03)
            prog.progress(p)
        status.empty()
        prog.empty()

        # Prediction
        try:
            X = np.array(features).reshape(1, -1)
            pred = model.predict(X)[0]
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(X)[0][1])
            elif hasattr(model, "decision_function"):
                score = model.decision_function(X)[0]
                proba = float(1 / (1 + np.exp(-score)))
            else:
                proba = float(pred)
            prob_pct = int(round(proba * 100))

            # Store prediction in session_state
            st.session_state["last_prediction"] = int(pred)
            st.session_state["last_probability"] = float(proba)
            st.session_state["last_prob_pct"] = int(prob_pct)

            # render result immediately
            risk_label, risk_color = prob_to_risk(proba)
            result_html = f"""
                <div style="padding:8px;border-radius:6px;">
                    <p style="margin:2px 0;"><strong>Prediction:</strong> {'Malaria' if int(pred) == 1 else 'No Malaria'}</p>
                    <p style="margin:2px 0;"><strong>Probability:</strong> <span style="font-size:18px">{prob_pct}%</span></p>
                    <div style="margin-top:6px;"><span class="risk-badge" style="background:{risk_color}">{risk_label}</span></div>
                </div>
            """
            result_area.markdown(result_html, unsafe_allow_html=True)

            battery_html = f"""
            <div style="margin-top:8px;">
              <div class="battery">
                <div class="battery-inner" style="width:{prob_pct}%; background:{risk_color};">{prob_pct}%</div>
                <div class="battery-tip"></div>
              </div>
            </div>
            """
            battery_area.markdown(battery_html, unsafe_allow_html=True)

            if int(pred) == 1:
                advice_area.warning("Possible malaria. Please visit a hospital for diagnostic confirmation.")
            else:
                advice_area.success("Low likelihood of malaria. If symptoms persist, seek medical care.")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            log_debug("Prediction exception: " + traceback.format_exc())

# ----------------------
# If a previous prediction exists in session_state, show it so feedback can be submitted
# ----------------------
if st.session_state.get("last_prediction") is not None:
    # show the last saved prediction/result (so it's visible after rerun)
    last_pred = st.session_state["last_prediction"]
    last_prob_pct = st.session_state["last_prob_pct"]
    last_prob = st.session_state["last_probability"]
    risk_label, risk_color = prob_to_risk(last_prob)
    result_html = f"""
        <div style="padding:8px;border-radius:6px;">
            <p style="margin:2px 0;"><strong>Prediction:</strong> {'Malaria' if int(last_pred) == 1 else 'No Malaria'}</p>
            <p style="margin:2px 0;"><strong>Probability:</strong> <span style="font-size:18px">{last_prob_pct}%</span></p>
            <div style="margin-top:6px;"><span class="risk-badge" style="background:{risk_color}">{risk_label}</span></div>
        </div>
    """
    result_area.markdown(result_html, unsafe_allow_html=True)
    battery_html = f"""
    <div style="margin-top:8px;">
      <div class="battery">
        <div class="battery-inner" style="width:{last_prob_pct}%; background:{risk_color};">{last_prob_pct}%</div>
        <div class="battery-tip"></div>
      </div>
    </div>
    """
    battery_area.markdown(battery_html, unsafe_allow_html=True)

    if int(last_pred) == 1:
        advice_area.warning("Possible malaria. Please visit a healthcare facility for diagnostic confirmation.")
    else:
        advice_area.success("Low likelihood of malaria. If symptoms persist, seek medical care.")

    # Feedback form (submit inside form). Uses session_state values for prediction/probability.
    with feedback_area.container():
        with st.form("feedback_form"):
            st.write("Was this assessment helpful?")
            helpful = st.radio("", ["Select", "Yes", "No"], index=0, horizontal=True)
            st.write("Did you later confirm diagnosis at a clinic?")
            clinic_result = st.selectbox("", ["Select", "I tested positive (malaria)", "I tested negative (not malaria)", "I did not check"])
            comment = st.text_area("Comments (optional)", value="", max_chars=500)
            submit = st.form_submit_button("Submit feedback")

            if submit:
                # build record matching Supabase table columns
                rec = {
                    "probability": int(st.session_state["last_prob_pct"]),  # integer percent
                    "prediction": "Malaria" if int(st.session_state["last_prediction"]) == 1 else "No Malaria",
                    "helpful": helpful if helpful != "Select" else "",
                    "clinic_result": clinic_result if clinic_result != "Select" else "",
                    "comment": comment,
                }

                # log and insert
                log_debug("Attempting to insert to Supabase. Payload: " + json.dumps(rec, ensure_ascii=False))
                success, resp = supabase_insert(rec)
                if success:
                    st.success("Thanks, your feedback was submitted.")
                    log_debug("Supabase insert succeeded: " + json.dumps(resp if isinstance(resp, dict) else {"resp": str(resp)}))
                else:
                    st.error("Could not submit feedback.")
                    log_debug("Supabase insert FAILED: " + json.dumps(resp if isinstance(resp, dict) else {"resp": str(resp)}))
                    # fallback save locally
                    st.session_state["feedback_list"].append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "probability_pct": st.session_state["last_prob_pct"],
                        "predicted": int(st.session_state["last_prediction"]),
                        "helpful": rec["helpful"],
                        "clinic_result": rec["clinic_result"],
                        "comment": rec["comment"],
                    })
                    st.info("Feedback saved locally for this session only.")

# Show local session feedback if present
if st.session_state["feedback_list"]:
    st.markdown("---")
    st.subheader("Recent feedback (this session)")
    st.dataframe(st.session_state["feedback_list"][::-1], height=200)

