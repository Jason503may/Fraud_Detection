import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main {background-color: #f0f2f6;}
    .stMetric {background-color: #ffffff; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .fraud-high {background-color: #fee; color: #dc2626;}
    .fraud-medium {background-color: #fef3c7; color: #d97706;}
    .fraud-low {background-color: #d1fae5; color: #059669;}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Financial Fraud Detection App")
st.markdown("---")

# Sidebar
st.sidebar.header("📊 Settings")
contamination = st.sidebar.slider("Fraud Ratio Estimate", 0.01, 0.2, 0.05, 0.01)
features = st.sidebar.multiselect("Select Features for Detection", [], help="Auto-detects numerical/categorical")

if st.sidebar.button("🔄 Detect Fraud", type="primary"):
    st.sidebar.success("Analysis started!")

# File uploader
uploaded_file = st.file_uploader("📁 Upload your CSV file", type="csv", help="Upload financial transaction CSV")

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)

    df = load_data(uploaded_file)
    st.success(f"✅ Loaded {len(df):,} transactions!")

    # Data preview
    with st.expander("👀 Preview Data", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df.head(), use_container_width=True)
        with col2:
            st.metric("Total Transactions", len(df))
            st.metric("Avg Amount", f"${df.get('amount', pd.Series([0])).mean():,.2f}")
            st.metric("Unique Users", df.get('user_id', pd.Series(range(len(df)))).nunique())

    if st.button("🚀 Run Fraud Detection", type="primary"):
        with st.spinner("Detecting fraud..."):
            # Preprocess
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()

            # Impute missing
            imputer_num = SimpleImputer(strategy='mean')
            df[num_cols] = imputer_num.fit_transform(df[num_cols])

            # Encode categorical
            le_dict = {}
            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le

            # Features for model (all except id cols)
            feature_cols = [col for col in df.columns if col not in ['transaction_id', 'user_id']]
            X = df[feature_cols].values

            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Model
            model = IsolationForest(contamination=contamination, random_state=42)
            preds = model.fit_predict(X_scaled)
            anomaly_scores = model.decision_function(X_scaled)
            scores_neg = -anomaly_scores  # Higher = more anomalous

            df['fraud'] = preds == -1
            df['anomaly_score'] = scores_neg
            df['risk_percentile'] = df['anomaly_score'].rank(pct=True) * 100
            df['risk_level'] = pd.cut(df['risk_percentile'], bins=[0, 70, 90, 100], labels=['low', 'medium', 'high'])

        # Results
        st.markdown("## 📈 Detection Results")
        col1, col2, col3, col4 = st.columns(4)
        fraud_pct = df['fraud'].mean() * 100
        high_risk = (df['risk_level'] == 'high').mean() * 100
        med_risk = (df['risk_level'] == 'medium').mean() * 100
        low_risk = (df['risk_level'] == 'low').mean() * 100

        with col1:
            st.metric("Fraud Detected", f"{fraud_pct:.1f}%", delta=f"{fraud_pct:.1f}%")
        with col2:
            st.markdown(f'<div class="fraud-high">High Risk: {high_risk:.1f}%</div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="fraud-medium">Medium Risk: {med_risk:.1f}%</div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="fraud-low">Low Risk: {low_risk:.1f}%</div>', unsafe_allow_html=True)

        # Charts
        col_a, col_b = st.columns(2)
        with col_a:
            fig_pie = px.pie(df, names='risk_level', title='Risk Distribution')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            if 'amount' in df.columns and 'time' in df.columns:
                fig_scatter = px.scatter(df, x='time', y='amount', color='risk_level', 
                                        title='Transactions: Amount vs Time by Risk')
                st.plotly_chart(fig_scatter, use_container_width=True)

        # Predictions table
        st.markdown("## 📋 Fraud Predictions")
        view = st.selectbox("View", ["All", "High Risk Only", "Fraud Only"])
        if view == "High Risk Only":
            filtered = df[df['risk_level'] == 'high']
        elif view == "Fraud Only":
            filtered = df[df['fraud'] == True]
        else:
            filtered = df
        st.dataframe(filtered[['fraud', 'risk_level', 'anomaly_score', 'risk_percentile']].round(4), use_container_width=True)

        # Download
        csv_buffer = BytesIO()
        filtered.to_csv(csv_buffer, index=False)
        st.download_button("💾 Download Results CSV", csv_buffer.getvalue(), "fraud_predictions.csv", "text/csv")

else:
    st.info("👆 Upload a CSV file to get started. Sample columns: transaction_id, amount, category, time, location, user_id.")

st.markdown("---")
st.caption("Built with Streamlit & scikit-learn. Upload your data and detect fraud instantly!")
