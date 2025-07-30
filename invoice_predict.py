import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\abhin\OneDrive\ドキュメント\invoice-payment-prediction\invoice-payment-prediction\improved_invoice_data.csv")

# Convert to datetime
df["invoice_date"] = pd.to_datetime(df["invoice_date"])
df["due_date"] = pd.to_datetime(df["due_date"])
df["actual_payment_date"] = pd.to_datetime(df["actual_payment_date"])

# Feature engineering
df["days_to_pay"] = (df["actual_payment_date"] - df["invoice_date"]).dt.days
df["days_until_due"] = (df["due_date"] - df["invoice_date"]).dt.days

# Remove outliers
q1 = df["days_to_pay"].quantile(0.01)
q99 = df["days_to_pay"].quantile(0.99)
df = df[(df["days_to_pay"] >= q1) & (df["days_to_pay"] <= q99)]

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=["job_title", "location"], drop_first=True)

# Features
base_features = ["monthly_income", "credit_score", "amount", "days_until_due"]
one_hot_cols = [col for col in df_encoded.columns if col.startswith("job_title_") or col.startswith("location_")]
feature_columns = base_features + one_hot_cols

# Classification
X_class = df_encoded[feature_columns]
y_class = df_encoded["payment_status"]

# Regression
X_reg = df_encoded[feature_columns]
y_reg = df_encoded["days_to_pay"]

# Split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Scaling
scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)

# Model training (NO CHANGES HERE)
clf = LogisticRegression()
clf.fit(X_train_c_scaled, y_train_c)
y_pred_c = clf.predict(X_test_c_scaled)
accuracy = accuracy_score(y_test_c, y_pred_c)

reg = RandomForestRegressor()
reg.fit(X_train_r_scaled, y_train_r)
y_pred_r = reg.predict(X_test_r_scaled)
r2 = r2_score(y_test_r, y_pred_r)

# Streamlit App Config
st.set_page_config(page_title="Invoice Payment Prediction", layout="centered")

# ✅ Background + text color match (NO layout or logic change)
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1556745757-8d76bdb6984b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }
    h1, h2, h3, h4, h5, h6, label, .st-bb, .st-c3, .st-c4, .st-c5, .st-c6, .st-ce {
        color: white !important;
        text-shadow: 1px 1px 2px black;
    }
    .stSidebar {
        background-color: rgba(0, 0, 0, 0.6);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with Accuracy
st.sidebar.title("Model Accuracy")
st.sidebar.metric("Classification Accuracy", f"{accuracy:.2f}")
st.sidebar.metric("Regression R² Score", f"{r2:.2f}")

st.title("📄 Invoice Payment Prediction App")

with st.form("user_input"):
    st.subheader("Enter Invoice Details")
    invoice_id = st.text_input("Invoice ID")
    person_name = st.text_input("Person Name")
    job_title = st.selectbox("Job Title", df["job_title"].unique())
    monthly_income = st.number_input("Monthly Income", min_value=0)
    credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=600)
    invoice_date = st.date_input("Invoice Date")
    due_date = st.date_input("Due Date")
    amount = st.number_input("Amount", min_value=0)
    location = st.selectbox("Location", df["location"].unique())

    submitted = st.form_submit_button("Predict Payment")

    if submitted:
        days_until_due = (due_date - invoice_date).days

        user_df = pd.DataFrame({
            "monthly_income": [monthly_income],
            "credit_score": [credit_score],
            "amount": [amount],
            "days_until_due": [days_until_due]
        })

        for col in one_hot_cols:
            if col == f"job_title_{job_title}":
                user_df[col] = 1
            elif col == f"location_{location}":
                user_df[col] = 1
            else:
                user_df[col] = 0

        for col in feature_columns:
            if col not in user_df.columns:
                user_df[col] = 0

        user_df = user_df[feature_columns]

        user_scaled_c = scaler_c.transform(user_df)
        user_scaled_r = scaler_r.transform(user_df)

        # Predict from model
        model_status = clf.predict(user_scaled_c)[0]
        model_days = int(reg.predict(user_scaled_r)[0])
        expected_payment_date = invoice_date + pd.to_timedelta(model_days, unit='D')

        # ✅ Post-prediction manual override (ONLY for edge case)
        if monthly_income == 0 and amount > 5000:
            final_status = 0  # Force "No"
        else:
            final_status = model_status  # Use model prediction

        # Show result
        st.success("Prediction Completed")
        st.write(f"### ✅ Will Pay on Time: {'Yes' if final_status == 1 else 'No'}")
        st.write(f"### ⏳ Expected Days to Pay: {model_days} days")
        st.write(f"### 📅 Expected Payment Date: {expected_payment_date}")
