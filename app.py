# app.py
import os
import traceback
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "dataset", "creditcard.csv")
    if not os.path.exists(csv_path):
        alt_path = os.path.join(base_dir, "data", "creditcard.csv")
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found. Tried:\n  {csv_path}\n  {alt_path}")
    df = pd.read_csv(csv_path)
    return df

try:
    data = load_data()
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Use only a sample for performance
    max_rows = st.sidebar.number_input("Max rows to use (for demo)", min_value=500, max_value=200000, value=10000, step=500)
    if data.shape[0] > max_rows:
        data_sample = data.sample(max_rows, random_state=42).reset_index(drop=True)
        st.info(f"Using a random sample of {max_rows} rows for faster demo.")
    else:
        data_sample = data.copy()

    # Prepare data
    X = data_sample.drop('Class', axis=1)
    y = data_sample['Class']

    # Scale Time and Amount
    scaler = StandardScaler()
    to_scale = [c for c in ['Time', 'Amount'] if c in X.columns]
    if to_scale:
        X[to_scale] = scaler.fit_transform(X[to_scale])

    # Train-test split
    test_size = st.sidebar.slider("Test set fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # Model training
    n_estimators = st.sidebar.slider("n_estimators", min_value=10, max_value=200, value=50, step=10)
    max_depth = st.sidebar.slider("max_depth (None=0)", min_value=0, max_value=50, value=0, step=1)
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, max_depth=(None if max_depth==0 else max_depth))

    with st.spinner("Training model..."):
        rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    st.write("### Model Evaluation")
    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{acc:.4f}")

    st.write("Classification Report")
    st.text(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Feature importances
    if hasattr(rf, "feature_importances_"):
        fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
        st.write("### Top 10 Feature Importances")
        st.bar_chart(fi)

except Exception as e:
    st.error("An error occurred. See details below.")
    st.exception(traceback.format_exc())
    print(traceback.format_exc())
