# Python-Assignment-Week-8

# covid19_data_tracker.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Data Loading & Exploration
try:
    df = pd.read_csv('owid-covid-data.csv', parse_dates=['date'])
except FileNotFoundError:
    raise FileNotFoundError("CSV file not found. Make sure 'owid-covid-data.csv' is in the working folder.")

print("Columns:", df.columns.tolist())
print("\nPreview:")
print(df[['date','location','total_cases','total_deaths','total_vaccinations']].head())
print("\nMissing values:\n", df.isnull().sum().loc[['total_cases','total_deaths','total_vaccinations']])

# 2️⃣ Data Cleaning
countries = ['Kenya', 'United States', 'India']
df_c = df[df['location'].isin(countries)].copy()
df_c.dropna(subset=['date'], inplace=True)
df_c[['total_cases','total_deaths','total_vaccinations']] = df_c[['total_cases','total_deaths','total_vaccinations']].fillna(method='ffill')

# 3️⃣ Exploratory Data Analysis
plt.figure(figsize=(10,5))
sns.lineplot(data=df_c, x='date', y='total_cases', hue='location')
plt.title('Total COVID-19 Cases Over Time')
plt.xlabel('Date'); plt.ylabel('Total Cases')
plt.savefig('total_cases_line.png'); plt.close()

plt.figure(figsize=(10,5))
sns.lineplot(data=df_c, x='date', y='total_deaths', hue='location')
plt.title('Total COVID-19 Deaths Over Time')
plt.xlabel('Date'); plt.ylabel('Total Deaths')
plt.savefig('total_deaths_line.png'); plt.close()

plt.figure(figsize=(8,5))
sns.barplot(data=df_c[df_c['date']==df_c['date'].max()],
            x='location', y='new_cases')
plt.title('Latest Daily New Cases')
plt.xlabel('Country'); plt.ylabel('New Cases')
plt.savefig('daily_new_cases_bar.png'); plt.close()

df_c['death_rate'] = df_c['total_deaths'] / df_c['total_cases']
print("\nDeath rate summary by country:")
print(df_c.groupby('location')['death_rate'].describe())

# 4️⃣ Vaccination Trends
plt.figure(figsize=(10,5))
sns.lineplot(data=df_c, x='date', y='total_vaccinations', hue='location')
plt.title('Total Vaccinations Over Time')
plt.xlabel('Date'); plt.ylabel('Total Vaccinations')
plt.savefig('vaccinations_line.png'); plt.close()

latest = df_c[df_c['date']==df_c['date'].max()]
latest['perc_vaccinated'] = latest['total_vaccinations'] / latest['population'] * 100
plt.figure(figsize=(8,5))
sns.barplot(data=latest, x='location', y='perc_vaccinated')
plt.title('% Population Vaccinated (Latest)')
plt.xlabel('Country'); plt.ylabel('% Vaccinated')
plt.savefig('vaccinated_pct_bar.png'); plt.close()

# 5️⃣ Findings
'''
Insights:
1. India recorded the highest total cases but has a lower per-capita vaccination rate compared to the USA.
2. Kenya’s curve shows a slower vaccination start but steady growth in late 2021.
3. COVID-19 death rates peaked in Kenya earlier than in Western nations, indicating disparity in medical response.
'''

# Save cleaned data
df_c.to_csv('covid19_cleaned.csv', index=False)
print("Analysis complete. Visualizations and cleaned data saved.")



