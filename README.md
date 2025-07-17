# Python-Assignment-Week-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Introduce a missing value for demonstration (simulate error handling)
df.loc[0, 'sepal length (cm)'] = np.nan

# Handle missing values by filling with the mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Data exploration
print("First 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# Basic statistics
print("\nDescriptive Statistics:")
print(df.describe())

print("\nMean Values Grouped by Species:")
print(df.groupby('species').mean(numeric_only=True))

# Create a dummy time column to simulate trend visualization
df['Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')

# Set plotting style
sns.set(style="whitegrid")

# Line chart - Sepal length over time
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['sepal length (cm)'], label='Sepal Length')
plt.title('Sepal Length Over Time')
plt.xlabel('Date')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.tight_layout()
plt.savefig('line_chart.png')
plt.close()

# Bar chart - Average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='species', y='petal length (cm)')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.savefig('bar_chart.png')
plt.close()

# Histogram - Sepal width
plt.figure(figsize=(8, 5))
plt.hist(df['sepal width (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('histogram.png')
plt.close()

# Scatter plot - Sepal length vs Petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.tight_layout()
plt.savefig('scatter_plot.png')
plt.close()
