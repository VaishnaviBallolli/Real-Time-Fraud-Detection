import time
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Real-Time Fraud Detection", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_stream():
    return pd.read_csv("demo_stream.csv")

model = load_model()
data = load_stream()
FEATURES = list(model.feature_names_in_)

def score(row):
    x = row[FEATURES].to_frame().T.astype(float)
    return model.predict_proba(x)[0, 1]