import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
import joblib
import sklearn



df_filtered = pd.read_csv('cleaned_for_python.csv')
df_filtered = df_filtered[['Weight.x','Height','Converted_Arm','Converted_Hand','Position_Group']]
bayesian_pipeline = joblib.load("bayesian_ridge_model.pkl")  # Update with actual path
features = joblib.load("model_features.pkl")

st.sidebar.header("Filters")

# Add 'All' option to position filter
position_options = sorted(
    [pos for pos in df_filtered["Position_Group"].dropna().unique() if pos not in ["DL", "OL"]]
)

df_filtered.columns = ['HS_Weight','Height','Arm_Length','Hand_Size','Position_Group']

# === User Inputs ===
st.sidebar.header("Enter Player Metrics")
position = st.sidebar.selectbox("Position", position_options)
height = st.sidebar.number_input("Height (format: 6003 = 6'0\"3)", min_value=50.0, max_value=7000.0, step= .125)
hand_size = st.sidebar.number_input("Hand Size (inches)", min_value=5.0, max_value=13.0, step=0.125)
arm_length = st.sidebar.number_input("Arm Length (inches)", min_value=10.0, max_value=90.0, step=0.125)
hs_weight = st.sidebar.number_input("High School Weight", min_value=0.0, max_value=400.0, step=0.125)


df_filtered = df_filtered[df_filtered["Position_Group"] == position]
metrics = {
    "Height": height,
    "Hand_Size": hand_size,
    "HS_Weight": hs_weight,
    "Arm_Length": arm_length,
    
}

def predict_college_weight(HS_Weight, Height, Converted_Hand, Converted_Arm, Position_Group):
    # Create input DataFrame
    input_df = pd.DataFrame({
        'HS_Weight': [HS_Weight],
        'Height': [Height],
        'Converted_Hand': [Converted_Hand],
        'Converted_Arm': [Converted_Arm],
        'Position_Group': [Position_Group]
    })

    # Add interaction terms
    for group in df_filtered['Position_Group'].unique():
        input_df[f'HS_Weight_x_{group}'] = HS_Weight if Position_Group == group else 0

    # Add missing columns with 0
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[features]

    # Predict with interval
    prediction, std = bayesian_pipeline.predict(input_df, return_std=True)
    lower = prediction[0] - 1.96 * std[0]
    upper = prediction[0] + 1.96 * std[0]

    return prediction[0], lower, upper

# === Run Prediction ===
if st.sidebar.button("Predict College Weight"):
    pred, low, high = predict_college_weight(hs_weight, height, hand_size, arm_length, position)
    st.markdown(f"### 📊 Predicted College Weight: **{pred:.2f} lbs**")
    st.markdown(f"95% Prediction Interval: **({low:.2f}, {high:.2f}) lbs**")



# UCLA colors
UCLA_BLUE = "#2774AE"
UCLA_GOLD = "#FFD100"

# Plotting function
def plot_percentile(data, value, metric):
    percentile = percentileofscore(data, value)
    fig, ax = plt.subplots()
    sns.kdeplot(data, fill=True, color=UCLA_GOLD, ax=ax)
    ax.axvline(value, color=UCLA_BLUE, linestyle="--")
    ax.text(value + 0.2, ax.get_ylim()[1] * 0.05, f"{value} \n{percentile:.1f}th pct", color=UCLA_BLUE, fontweight='bold')
    ax.set_title(f"{metric} Distribution")
    ax.set_xlabel(metric)
    ax.set_ylabel("Density")
    return fig

# Show plots in a 2x2 grid
st.subheader("Percentile Visualizations")

cols = st.columns(2)
metric_names = list(metrics.keys())

for i, metric in enumerate(metric_names):
    value = metrics[metric]
    with cols[i % 2]:
        if metric in df_filtered.columns and df_filtered[metric].notna().sum() > 0:
            fig = plot_percentile(df_filtered[metric].dropna(), value, metric)
            st.pyplot(fig)
        else:
            st.write(f"{metric} not found or insufficient data.")

