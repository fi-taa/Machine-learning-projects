import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
data = pd.read_csv('../data/train.csv')

# Data Overview
print(data.head())

# Data Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data['Age'].dropna(), bins=20, kde=True)  # Exclude rows with missing 'Age'
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data['Fare'], bins=30, kde=True)
plt.xlabel('Fare')
plt.ylabel('Count')
plt.title('Fare Distribution')
plt.show()

# Missing Data Visualization
plt.figure(figsize=(10, 5))
sns.heatmap(data.isnull(), cmap='viridis', yticklabels=False, cbar=False)
plt.title('Missing Data')
plt.show()

# Select numeric columns for the correlation matrix
numeric_columns = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
corr_matrix = data[numeric_columns].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# Survival vs. Features
plt.figure(figsize=(12, 6))
sns.barplot(x='Sex', y='Survived', data=data, ci=None)
plt.xlabel('Sex')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Gender')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Pclass', y='Survived', data=data, ci=None)
plt.xlabel('Pclass')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Passenger Class')
plt.show()

# Age Distribution by Class
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Age', data=data)
plt.xlabel('Pclass')
plt.ylabel('Age')
plt.title('Age Distribution by Passenger Class')
plt.show()

# Fare Distribution by Class
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Fare', data=data)
plt.xlabel('Pclass')
plt.ylabel('Fare')
plt.title('Fare Distribution by Passenger Class')
plt.show()

# Survival by Age and Gender
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Survived', hue='Sex', data=data)
plt.xlabel('Age')
plt.ylabel('Survived')
plt.title('Survival by Age and Gender')
plt.show()

# Survival by Class and Embarked
plt.figure(figsize=(12, 6))
sns.barplot(x='Pclass', y='Survived', hue='Embarked', data=data, ci=None)
plt.xlabel('Pclass')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Passenger Class and Embarked')
plt.show()
