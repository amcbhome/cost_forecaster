import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Cost Forecasting Tool", layout="centered")

st.title("Production Cost Forecast")

st.markdown("""
**Linear regression analysis of the correlation between activity and cost [Source](https://www.accaglobal.com/uk/en/student/exam-support-resources/fundamentals-exams-study-resources/f5/technical-articles/regression.html)**
""")

# Step 1
st.subheader("Step 1: Upload or Enter Historical Data")
uploaded_file = st.file_uploader("Upload CSV with columns: x, y", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame({
        "x": [15, 45, 25, 55, 30, 20, 35, 60],
        "y": [300, 615, 470, 680, 520, 350, 590, 740]
    })

# Display table with custom column labels
display_df = df.rename(columns={
    "x": "Activity level",
    "y": "Production cost"
})
st.dataframe(display_df)

# Calculations
n = len(df)
sum_x = df["x"].sum()
sum_y = df["y"].sum()
sum_xy = (df["x"] * df["y"]).sum()
sum_x2 = (df["x"] ** 2).sum()
sum_y2 = (df["y"] ** 2).sum()

mean_x = sum_x / n
mean_y = sum_y / n

b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
a = mean_y - b * mean_x
r = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))

# Step 2: Show regression outputs
st.subheader("Step 2: Regression Results")
st.metric("Intercept (a)", f"{a:.2f}")
st.metric("Slope (b)", f"{b:.2f}")
st.metric("Correlation (r)", f"{r:.4f}")

# Step 3: Forecasting
st.subheader("Step 3: Forecast Cost")
input_x = st.number_input("Enter projected activity level (x)", min_value=0, step=1)

if input_x:
    y_pred = a + b * input_x
    st.success(f"Estimated production cost: **{y_pred:.2f}**")
