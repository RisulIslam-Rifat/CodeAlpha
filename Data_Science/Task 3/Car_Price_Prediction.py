# =====================================================
# CAR PRICE PREDICTION REGRESSION MODEL
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------
df = pd.read_csv("/mnt/data/car data.csv")

print("First rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# -----------------------------------------------------
# 2. DATA PREPROCESSING
# -----------------------------------------------------

# Rename columns for easy handling
df.columns = df.columns.str.replace(" ", "_")

# Identify categorical & numerical features
categorical_features = ["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"]
numeric_features = ["Present_Price", "Kms_Driven", "Owner", "Year"]

# Target variable
target = "Selling_Price"

# -----------------------------------------------------
# 3. FEATURE ENGINEERING
# -----------------------------------------------------

df["Car_Age"] = 2024 - df["Year"]   # convert Year to car age
numeric_features.append("Car_Age")
numeric_features.remove("Year")     # remove original year feature

# -----------------------------------------------------
# 4. TRAIN–TEST SPLIT
# -----------------------------------------------------
X = df[categorical_features + numeric_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------
# 5. PREPROCESSING + MODEL PIPELINE
# -----------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# -----------------------------------------------------
# 6. TRAIN THE MODEL
# -----------------------------------------------------
model.fit(X_train, y_train)

# -----------------------------------------------------
# 7. PREDICT & EVALUATE
# -----------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📌 MODEL PERFORMANCE")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# -----------------------------------------------------
# 8. VISUALIZATION: Actual vs Predicted
# -----------------------------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.show()

# -----------------------------------------------------
# 9. VISUALIZATION: Price Distribution
# -----------------------------------------------------
plt.figure(figsize=(8,6))
sns.histplot(df["Selling_Price"], kde=True)
plt.title("Distribution of Selling Price")
plt.show()

print("\n🎉 Analysis Complete!")
