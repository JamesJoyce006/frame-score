# Metric_Dashboard_v2.py

# streamlit: title = Custom Tab Title


import streamlit as st
import pandas as pd

st.title("Information About Model")


st.subheader("What is Ridge Regression?")
# Sample placeholder content
st.sidebar.image("/Users/jamesjoyce/Metric_Dashboard_v2/ucla_logo.png", use_container_width=True)



# Bullet Point Explanation
ridge_explanation = [
    "Ridge regression is a type of model used to make predictions based on data.",
    "It’s helpful when you have several factors (like height, weight, age) that all affect the outcome.",
    "Regular models can overfit — meaning they try too hard to match the data and make bad predictions on new data.",
    "Ridge regression adds a small penalty to prevent the model from making extreme or overly complex rules.",
    "This penalty helps the model stay balanced and more accurate when used on new people or situations.",
    "Ridge regression is commonly used when you want more stable and reliable predictions."
]

# Display bullets
for point in ridge_explanation:
    st.markdown(f"- {point}")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.subheader('Evaluation Metrics of the Model')

coefficients = pd.read_csv('/Users/jamesjoyce/Metric_Dashboard_v2/Ridge_Regression_FM_Coefficients.csv',index_col=0)

rr_summary = pd.read_csv('/Users/jamesjoyce/Metric_Dashboard_v2/Ridge_Regression_FM_Eval.csv',index_col=0)

interval_data =  pd.read_csv('/Users/jamesjoyce/Metric_Dashboard_v2/Frame_Score_Interval_Data.csv')


col1, col2 = st.columns(2)

# Column 1
with col1:
    st.subheader("Coefficients")
    st.dataframe(coefficients)

# Column 2
with col2:
    st.subheader("Model Evaluation Metrics")
    st.dataframe(rr_summary)


import plotly.graph_objects as go
import numpy as np
import streamlit as st

# Example: df_preds = pd.read_csv("your_predictions.csv")

# x-axis is just the index (0, 1, 2, ...)
x = interval_data.index
predicted = interval_data["Predicted"]
lower = interval_data["Lower Bound (95%)"]
upper = interval_data["Upper Bound (95%)"]
actual = interval_data["Actual"]


# 45-degree reference line
min_val = min(actual.min(), predicted.min())
max_val = max(actual.max(), predicted.max())

fig = go.Figure()

# Actual vs. Predicted points
fig.add_trace(go.Scatter(
    x=actual,
    y=predicted,
    mode='markers',
    name='Predicted vs. Actual',
    marker=dict(color='blue', size=8),
    hovertemplate='Actual: %{x}<br>Predicted: %{y}<extra></extra>'
))

# Diagonal line y = x
fig.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    name='Ideal Prediction',
    line=dict(color='green', dash='dash')
))

fig.update_layout(
    title="📈 Actual vs. Predicted",
    xaxis_title="Actual",
    yaxis_title="Predicted",
    showlegend=True,
    template="plotly_white",
    width=700,
    height=500
)

st.plotly_chart(fig, use_container_width=True)
