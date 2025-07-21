import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
import joblib




df = pd.read_csv('cleaned_for_python.csv')
df = df[['Weight.x','Height','Converted_Arm','Converted_Hand','Position_Group']]

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = df.rename(columns={'Weight.x': 'College_Weight', 'Weight.y': 'HS_Weight'})
df = df[~df['Position_Group'].isin(['OL', 'DL'])]
df = df.fillna(df.mean(numeric_only=True))
df['Position_Group'] = df['Position_Group'].astype('category')
print(df.head())
# Ridge Regression Model to Predict College Weight from HS Weight and Other Features

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

# === 1. Load and Prepare Data ===
# Assume df is your existing DataFrame
# df = pd.read_csv('your_data.csv') or similar

# Center the continuous predictor


# Create interaction terms manually
for group in df['Position_Group'].unique():
    df[f'HS_Weight_x_{group}'] = df['HS_Weight'] * (df['Position_Group'] == group).astype(int)

# Define features and target
interaction_terms = [f'HS_Weight_x_{group}' for group in df['Position_Group'].unique()]
features = ['HS_Weight', 'Height', 'Converted_Hand', 'Converted_Arm'] + interaction_terms
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
lower_bound = y_pred - 1.96 * y_std
upper_bound = y_pred + 1.96 * y_std

interval_df = pd.DataFrame({
    'Predicted': y_pred,
    'Lower Bound (95%)': lower_bound,
    'Upper Bound (95%)': upper_bound,
    'Actual': y_test.reset_index(drop=True)
})

print(interval_df.head())

# === 8. Residual Plot ===
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, linestyle='--', color='red')
plt.xlabel("Predicted College Weight")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.tight_layout()
plt.show()
