import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load your dataset
df = pd.read_csv("WineQT.csv")

# Drop unnecessary column (if applicable)
df.drop(columns=['Id'], inplace=True)

# Define features and target variable
X = df.drop(columns=['quality'])
y = df['quality']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model (use your preferred model here)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "my_custom_model.pkl")
joblib.dump(scaler, "my_custom_scaler.pkl")

print("Model and Scaler saved successfully!")
