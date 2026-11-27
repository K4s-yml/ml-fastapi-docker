import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Pulsar Star Classifier", layout="centered")

st.title("Pulsar Star Prediction")
st.write("Enter radio signal parameters (HTRU2 dataset) to classify the candidate.")

if "features" not in st.session_state:
    st.session_state.features = [140.5625, 55.6838, -0.2346, -0.6996, 3.1998, 19.1104, 7.9755, 74.2422]

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("Example: Non-Pulsar (0)", use_container_width=True):
        st.session_state.features = [140.5625, 55.6838, -0.2346, -0.6996, 3.1998, 19.1104, 7.9755, 74.2422]
with col_btn2:
    if st.button("Example: Pulsar Star (1)", use_container_width=True):
        st.session_state.features = [99.3672, 41.5722, 1.5472, 4.1541, 27.5552, 61.7190, 2.2088, 3.6627]

vals = st.session_state.features

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        mean_ip = st.number_input("Mean of integrated profile", value=float(vals[0]), format="%.4f")
        std_ip = st.number_input("Std of integrated profile", value=float(vals[1]), format="%.4f")
        kurt_ip = st.number_input("Kurtosis of integrated profile", value=float(vals[2]), format="%.4f")
        skew_ip = st.number_input("Skewness of integrated profile", value=float(vals[3]), format="%.4f")

    with col2:
        mean_dm = st.number_input("Mean of DM-SNR curve", value=float(vals[4]), format="%.4f")
        std_dm = st.number_input("Std of DM-SNR curve", value=float(vals[5]), format="%.4f")
        kurt_dm = st.number_input("Kurtosis of DM-SNR curve", value=float(vals[6]), format="%.4f")
        skew_dm = st.number_input("Skewness of DM-SNR curve", value=float(vals[7]), format="%.4f")

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    payload = {
        "mean_integrated_profile": mean_ip,
        "std_integrated_profile": std_ip,
        "kurtosis_integrated_profile": kurt_ip,
        "skewness_integrated_profile": skew_ip,
        "mean_dm_snr_curve": mean_dm,
        "std_dm_snr_curve": std_dm,
        "kurtosis_dm_snr_curve": kurt_dm,
        "skewness_dm_snr_curve": skew_dm,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        pred = result["prediction"]
        prob = result["probability"]

        if pred == 1:
            st.success(f"Prediction: Pulsar Star (Class 1) — Probability: {prob:.2%}")
        else:
            st.info(f"Prediction: Non-Pulsar (Class 0) — Pulsar Probability: {prob:.2%}")

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {e}")