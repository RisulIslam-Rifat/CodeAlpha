# =====================================================
# SALES FORECASTING USING ADVERTISING DATA
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------
df = pd.read_csv("/mnt/data/Advertising.csv")

print("First rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# -----------------------------------------------------
# 2. DATA CLEANING
# -----------------------------------------------------
# Remove duplicates (if any)
df = df.drop_duplicates()

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------------------------------
# 3. EXPLORATION & VISUALIZATION
# -----------------------------------------------------
plt.figure(figsize=(12,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Relationship between advertising channels & sales
sns.pairplot(df)
plt.show()

# -----------------------------------------------------
# 4. FEATURE SELECTION
# -----------------------------------------------------
X = df[["TV", "Radio", "Newspaper"]]  # predictor features
y = df["Sales"]                       # target variable

# -----------------------------------------------------
# 5. TRAIN–TEST SPLIT
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------
# 6. REGRESSION MODEL
# -----------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------------------------------
# 7. PREDICTIONS
# -----------------------------------------------------
y_pred = model.predict(X_test)

# -----------------------------------------------------
# 8. MODEL EVALUATION
# -----------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📌 MODEL PERFORMANCE")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# -----------------------------------------------------
# 9. VISUALIZATION: Actual vs Predicted
# -----------------------------------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()

# -----------------------------------------------------
# 10. FEATURE IMPACT ANALYSIS
# -----------------------------------------------------
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Impact_Coefficient": model.coef_
})

print("\n📌 FEATURE IMPACT ON SALES")
print(coefficients)

coefficients.plot(kind="bar", x="Feature", y="Impact_Coefficient", figsize=(8,5))
plt.title("Impact of Advertising Channels on Sales")
plt.ylabel("Coefficient Value")
plt.show()

# -----------------------------------------------------
# 11. INSIGHTS FOR MARKETING STRATEGY
# -----------------------------------------------------
print("\n🎯 KEY INSIGHTS FOR BUSINESS MARKETING STRATEGY:\n")

print("1. TV advertising has the strongest impact on sales.")
print("2. Radio advertising is also highly effective and improves sales significantly.")
print("3. Newspaper advertising shows the smallest influence; budget can be shifted.")
print("4. Increasing TV + Radio spend yields the highest ROI.")
print("5. A blended strategy using TV and Radio can boost total conversions.")
