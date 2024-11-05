import pandas as pd
df = pd.read_csv('Downloads/USvideos.csv')
print(df)
print(df.info())
print(df.describe())
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram for each numerical column
df.hist(bins=15, figsize=(15, 10), color='skyblue')
plt.suptitle('Distribution of Numerical Variables')
plt.show()

# Boxplot for each numerical column to spot outliers
plt.figure(figsize=(15, 6))
sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']))
plt.title('Boxplot of Numerical Variables')
plt.show()
from scipy.stats import zscore
outliers = (df.select_dtypes(include=['float64', 'int64'])
              .apply(zscore)
              .abs() > 3)  # Z-score threshold for identifying outliers
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()
for col in df.select_dtypes(include='object'):
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=col, palette='viridis')
    plt.title(f'Distribution of {col}')
    plt.show()
