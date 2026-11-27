
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}
h1 {
    color: #172554;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("data/demo_stream.csv")


try:
    model = load_model()
    data = load_data()

except Exception as e:
    st.error(f"Unable to load model or dataset: {e}")
    st.stop()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if hasattr(model, "feature_names_in_"):
    FEATURES = list(model.feature_names_in_)
else:
    FEATURES = list(data.columns)

missing_columns = [col for col in FEATURES if col not in data.columns]

if missing_columns:
    st.error(f"Missing model features: {missing_columns}")
    st.stop()

X = data[FEATURES].apply(pd.to_numeric, errors="coerce")

if X.isnull().any().any():
    st.error("Dataset contains missing or non-numeric feature values.")
    st.stop()

try:
    probabilities = model.predict_proba(X)[:, 1]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🛡️ Fraud Detection")
st.sidebar.markdown("---")

threshold = st.sidebar.slider(
    "Fraud Detection Threshold",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard analyses sample credit card transactions "
    "using a trained XGBoost machine learning model."
)

st.sidebar.caption("Model: XGBoost")


# --------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------

results = data.copy()

results["Fraud Probability"] = probabilities

results["Prediction"] = (
    results["Fraud Probability"] >= threshold
).map({
    True: "Fraud",
    False: "Legitimate"
})

results["Risk Level"] = pd.cut(
    results["Fraud Probability"],
    bins=[-0.001, 0.30, 0.70, 1.001],
    labels=["Low", "Medium", "High"]
)


# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------

st.title("🛡️ Credit Card Fraud Detection")
st.markdown(
    "### Real-Time Fraud Monitoring Dashboard"
)

st.markdown(
    "Machine Learning Based Credit Card Transaction Analysis"
)

st.markdown("---")


# --------------------------------------------------
# KEY PERFORMANCE INDICATORS
# --------------------------------------------------

total_transactions = len(results)

fraud_transactions = (
    results["Prediction"] == "Fraud"
).sum()

legitimate_transactions = (
    results["Prediction"] == "Legitimate"
).sum()

fraud_percentage = (
    fraud_transactions / total_transactions * 100
    if total_transactions > 0 else 0
)

average_risk = probabilities.mean() * 100 if total_transactions else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Predicted Fraud",
    f"{fraud_transactions:,}"
)

col3.metric(
    "Legitimate Transactions",
    f"{legitimate_transactions:,}"
)

col4.metric(
    "Fraud Percentage",
    f"{fraud_percentage:.2f}%"
)

st.markdown("---")


# --------------------------------------------------
# CHARTS
# --------------------------------------------------

st.subheader("📊 Fraud Analytics")

chart1, chart2 = st.columns(2)

with chart1:

    st.markdown("#### Fraud vs Legitimate Transactions")

    count_data = pd.DataFrame({
        "Transaction Type": ["Fraud", "Legitimate"],
        "Count": [
            fraud_transactions,
            legitimate_transactions
        ]
    })

    fig1 = px.pie(
        count_data,
        names="Transaction Type",
        values="Count",
        hole=0.55,
        color="Transaction Type",
        color_discrete_map={
            "Fraud": "#ef4444",
            "Legitimate": "#16a34a"
        }
    )

    fig1.update_layout(
        margin=dict(t=20, b=20, l=20, r=20)
    )

    st.plotly_chart(fig1, use_container_width=True)


with chart2:

    st.markdown("#### Transaction Risk Distribution")

    risk_counts = (
        results["Risk Level"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .reset_index()
    )

    risk_counts.columns = ["Risk Level", "Count"]

    fig2 = px.bar(
        risk_counts,
        x="Risk Level",
        y="Count",
        color="Risk Level",
        color_discrete_map={
            "Low": "#16a34a",
            "Medium": "#f59e0b",
            "High": "#ef4444"
        },
        text="Count"
    )

    fig2.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20)
    )

    st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------
# FRAUD PROBABILITY DISTRIBUTION
# --------------------------------------------------

st.markdown("---")
st.subheader("📈 Fraud Probability Distribution")

fig3 = px.histogram(
    results,
    x="Fraud Probability",
    nbins=30,
    title="Distribution of Predicted Fraud Probabilities",
    color_discrete_sequence=["#6366f1"]
)

fig3.update_layout(
    xaxis_title="Fraud Probability",
    yaxis_title="Number of Transactions"
)

st.plotly_chart(fig3, use_container_width=True)


# --------------------------------------------------
# TRANSACTION DETAILS
# --------------------------------------------------

st.markdown("---")
st.subheader("🔍 Transaction Details")

filter_option = st.selectbox(
    "Filter Transactions",
    ["All", "Fraud", "Legitimate"]
)

if filter_option != "All":
    filtered_data = results[
        results["Prediction"] == filter_option
    ]
else:
    filtered_data = results

display_columns = (
    ["Prediction", "Fraud Probability", "Risk Level"]
    + FEATURES
)

st.dataframe(
    filtered_data[display_columns].style.format({
        "Fraud Probability": "{:.2%}"
    }),
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"Showing {len(filtered_data)} of {total_transactions} transactions."
)


# --------------------------------------------------
# DOWNLOAD RESULTS
# --------------------------------------------------

st.markdown("---")
st.subheader("📥 Download Prediction Results")

csv = results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Fraud Detection Report",
    data=csv,
    file_name="fraud_detection_results.csv",
    mime="text/csv"
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    "<center>Credit Card Fraud Detection | "
    "Machine Learning Project | Built with Streamlit</center>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# MANUAL TRANSACTION PREDICTION
# --------------------------------------------------

st.markdown("---")

st.title("🔍 Predict a New Transaction")

st.write(
    "Enter transaction feature values below to check "
    "whether the transaction is potentially fraudulent."
)

st.info(
    "Note: V1 to V28 are anonymized PCA-transformed features "
    "from the original credit card dataset. They are not card "
    "numbers or personal details. Use values in the same format "
    "as your model's training data."
)

# Create input form
with st.form("fraud_prediction_form"):

    st.subheader("Enter Transaction Details")

    input_values = {}

    columns = st.columns(3)

    for i, feature in enumerate(FEATURES):

        with columns[i % 3]:

            # Use sample data to suggest a default value
            default_value = float(data[feature].median())

            min_value = float(data[feature].min())
            max_value = float(data[feature].max())

            # Avoid identical minimum and maximum values
            if min_value == max_value:
                min_value -= 1
                max_value += 1

            input_values[feature] = st.number_input(
                label=feature,
                min_value=min_value,
                max_value=max_value,
                value=min(max(default_value, min_value), max_value),
                format="%.6f",
                key=f"input_{feature}"
            )

    submit = st.form_submit_button(
        "🚀 Predict Transaction",
        use_container_width=True
    )

# Make prediction
if submit:

    input_df = pd.DataFrame(
        [input_values],
        columns=FEATURES
    )

    prediction_probability = model.predict_proba(
        input_df
    )[0][1]

    predicted_class = (
        "Fraud"
        if prediction_probability >= threshold
        else "Legitimate"
    )

    st.markdown("---")
    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Fraud Probability",
            f"{prediction_probability * 100:.2f}%"
        )

    with result_col2:
        st.metric(
            "Prediction",
            predicted_class
        )

    if predicted_class == "Fraud":
        st.error(
            "⚠️ Potential Fraud Detected! "
            "This transaction has been flagged by the model."
        )
    else:
        st.success(
            "✅ Transaction Classified as Legitimate."
        )

    st.progress(float(prediction_probability))

    st.caption(
        f"Classification threshold: {threshold:.2f}"
    )

    st.download_button(
        "Download Transaction Details",
        data=input_df.to_csv(index=False),
        file_name="transaction_input.csv",
        mime="text/csv"
    )
