import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
import joblib

import streamlit_authenticator as stauth

# --- User credentials ---
users = {
    "james": {"name": "James Joyce", "password": "pass123", "email": "james@example.com"},
    "emma": {"name": "Emma Stone", "password": "hello456", "email": "emma@example.com"},
}

# --- Hash passwords ---
hashed_passwords = stauth.Hasher([users[u]["password"] for u in users]).generate()

# --- Setup authentication ---
credentials = {
    "usernames": {
        u: {"name": users[u]["name"], "password": pwd, "email": users[u]["email"]}
        for u, pwd in zip(users.keys(), hashed_passwords)
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "my_app",        # cookie name
    "abcdef",        # signature key (change to anything random for security)
    cookie_expiry_days=1
)

# --- Login widget ---
name, authentication_status, username = authenticator.login("Login", "main")

# --- Logic ---
if authentication_status:
    st.success(f"Welcome, {name}!")
    st.info(f"Logged in as: {credentials['usernames'][username]['email']}")

    # --- Place your app/dashboard here ---
    st.write("This is your protected dashboard.")

elif authentication_status is False:
    st.error("Username/password is incorrect")

elif authentication_status is None:
    st.warning("Please enter your username and password")


df = pd.read_csv('cleaned_for_python.csv')
df = df[['Weight.x','Weight.y','Height','Converted_Arm','Converted_Hand','Position_Group']]

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = df.rename(columns={'Weight.x': 'College_Weight', 'Weight.y': 'HS_Weight','Converted_Arm':'Arm_Length','Converted_Hand':'Hand_Size'})
df = df[~df['Position_Group'].isin(['OL', 'DL'])]
df = df.fillna(df.mean(numeric_only=True))
df['Position_Group'] = df['Position_Group'].astype('category')
print(df.head())
# Ridge Regression Model to Predict College Weight from HS Weight and Other Features

#
#
#





# Create interaction terms manually
for group in df['Position_Group'].unique():
    df[f'HS_Weight_x_{group}'] = df['HS_Weight'] * (df['Position_Group'] == group).astype(int)

# Define features and target
interaction_terms = [f'HS_Weight_x_{group}' for group in df['Position_Group'].unique()]
features = ['Height', 'HS_Weight','Hand_Size', 'Arm_Length'] + interaction_terms
X = df[features]
y = df['College_Weight']  # replace with the actual column name if different

# === 2. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Preprocessing and Pipeline ===
# All columns are numeric, so we only scale them
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), features)
])

bayesian_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('bayes_ridge', BayesianRidge())
])

# === 4. Fit Model ===
bayesian_pipeline.fit(X_train, y_train)

# === 5. Evaluate ===
y_pred, y_std = bayesian_pipeline.predict(X_test, return_std=True)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.2f}")
print(f"Test R^2: {r2:.3f}")

# === 6. Feature Importance ===
encoded_features = bayesian_pipeline.named_steps['preprocess'].get_feature_names_out()
coefficients = bayesian_pipeline.named_steps['bayes_ridge'].coef_

feature_importance = pd.DataFrame({
    'Feature': encoded_features,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', key=abs, ascending=False)

print(feature_importance)

# === 7. Prediction Intervals ===
lower_bound = y_pred - 1.7 * y_std
upper_bound = y_pred + 1.7 * y_std

interval_df = pd.DataFrame({
    'Predicted': y_pred,
    'Lower Bound (95%)': lower_bound,
    'Upper Bound (95%)': upper_bound,
    'Actual': y_test.reset_index(drop=True)
})

print(interval_df)








#
#
#

# === 1. UCLA Logo at the Top of Sidebar ===
st.sidebar.image("ucla_logo.png", use_container_width=True)

st.title("Frame Score Model")



# Sidebar filters
st.sidebar.header("Filters")

# Add 'All' option to position filter
position_options = sorted(
    [pos for pos in df["Position_Group"].dropna().unique() if pos not in ["DL", "OL"]]
)



# === User Inputs ===
st.sidebar.header("Enter Player Metrics")
position = st.sidebar.selectbox("Position", position_options)
height = st.sidebar.number_input("Height (format: 6003 = 6'0\"3)", min_value=50.0, max_value=7000.0, step= .125)
hand_size = st.sidebar.number_input("Hand Size (inches)", min_value=5.0, max_value=13.0, step=0.125)
arm_length = st.sidebar.number_input("Arm Length (inches)", min_value=10.0, max_value=90.0, step=0.125)
hs_weight = st.sidebar.number_input("High School Weight", min_value=0.0, max_value=400.0, step=0.125)


df_filtered = df[df["Position_Group"] == position]
metrics = {
    "Height": height,
    "Hand_Size": hand_size,
    "HS_Weight": hs_weight,
    "Arm_Length": arm_length,
    
}

def predict_college_weight(HS_Weight, Height, Hand_Size, Arm_Length, Position_Group):
    # Create input DataFrame
    input_df = pd.DataFrame({
        'HS_Weight': [HS_Weight],
        'Height': [Height],
        'Hand_Size': [Hand_Size],
        'Arm_Length': [Arm_Length],
        'Position_Group': [Position_Group]
    })

    # Add interaction terms
    for group in df['Position_Group'].unique():
        input_df[f'HS_Weight_x_{group}'] = HS_Weight if Position_Group == group else 0

    # Add missing columns with 0
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[features]

    # Predict with interval
    prediction, std = bayesian_pipeline.predict(input_df, return_std=True)
    lower = prediction[0] - 1.7 * std[0]
    upper = prediction[0] + 1.7 * std[0]

    return prediction[0], lower, upper

# === Run Prediction ===
if st.sidebar.button("Predict College Weight"):
    pred, low, high = predict_college_weight(hs_weight, height, hand_size, arm_length, position)
    st.markdown(f"### 📊 Predicted College Weight: **{pred:.2f} lbs**")
    st.markdown(f"Prediction Interval: **({low:.2f}, {high:.2f}) lbs**")



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
        if metric in df.columns and df[metric].notna().sum() > 0:
            fig = plot_percentile(df_filtered[metric].dropna(), value, metric)
            st.pyplot(fig)
        else:
            st.write(f"{metric} not found or insufficient data.")




