
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Credit Card Fraud Detection — Demo", layout="wide")

st.title("Credit Card Fraud Detection — Demo")
st.write("Upload transactions CSV with columns: V1..V28, Time, Amount.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # Show all rows scrollable
    st.subheader("Uploaded Data")
    st.dataframe(df, height=500)

    # Create a copy for output
    out_df = df.copy().reset_index(drop=True)

    # Add dummy predictions for testing
    out_df['probability_fraud'] = np.random.rand(len(out_df))
    out_df['xgb_pred'] = (out_df['probability_fraud'] >= 0.5).astype(int)
    out_df['isolation_anomaly'] = np.random.randint(0, 2, size=len(out_df))

    st.subheader("Predictions")
    st.dataframe(out_df, height=500)

    # Download button
    csv = out_df.to_csv(index=False)
    st.download_button("Download predictions CSV", csv, file_name="predictions.csv", mime="text/csv")

else:
    st.info("Upload a CSV to see predictions.")
