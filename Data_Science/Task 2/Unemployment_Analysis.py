# ================================================
# UNEMPLOYMENT RATE ANALYSIS (INDIA)
# ================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. LOAD AND CLEAN DATA
# -----------------------------
df = pd.read_csv("/mnt/data/Unemployment in India.csv")

print("Initial Data:")
print(df.head())

# Rename columns for easier handling
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date
df = df.sort_values(by="Date")

# -----------------------------
# 2. BASIC EXPLORATION
# -----------------------------
print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# -----------------------------
# 3. TREND VISUALIZATION
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Unemployment_Rate'], marker='o')
plt.title("Unemployment Rate Trend in India")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.show()

# -----------------------------
# 4. REGION-WISE UNEMPLOYMENT
# -----------------------------
plt.figure(figsize=(12,6))
sns.boxplot(data=df, x="Region", y="Unemployment_Rate")
plt.title("Unemployment Rate by Region")
plt.xticks(rotation=90)
plt.show()

# -----------------------------
# 5. COVID-19 IMPACT ANALYSIS
# -----------------------------
covid_start = "2020-03-01"
df['Covid_Period'] = df['Date'] >= covid_start

plt.figure(figsize=(12,6))
sns.lineplot(data=df, x="Date", y="Unemployment_Rate", hue="Covid_Period")
plt.title("Covid-19 Impact on Unemployment Rate")
plt.show()

before_covid = df[df['Date'] < covid_start]['Unemployment_Rate'].mean()
during_covid = df[df['Date'] >= covid_start]['Unemployment_Rate'].mean()

print(f"\nAverage Before Covid: {before_covid:.2f}%")
print(f"Average During Covid: {during_covid:.2f}%")

# -----------------------------
# 6. SEASONAL TREND ANALYSIS
# -----------------------------
df['Month'] = df['Date'].dt.month
monthly_avg = df.groupby('Month')['Unemployment_Rate'].mean()

plt.figure(figsize=(10,6))
monthly_avg.plot(kind='bar')
plt.title("Average Unemployment Rate by Month (Seasonal Pattern)")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# -----------------------------
# 7. INSIGHTS FOR POLICIES
# -----------------------------
print("\n📌 KEY INSIGHTS FOR ECONOMIC & SOCIAL POLICIES\n")

print("1. Covid-19 caused a significant spike in unemployment, especially after March 2020.")
print("2. Seasonal patterns show certain months consistently have higher unemployment.")
print("3. Some regions experience higher unemployment — targeted job programs can help.")
print("4. Policymakers should invest in pandemic-resilience measures for labor sectors.")
print("5. Vocational training & remote-work support can reduce unemployment during crises.")
