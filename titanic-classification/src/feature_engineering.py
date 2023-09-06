import pandas as pd

# Load the training dataset
train_data = pd.read_csv('train.csv')

# Load the test dataset
test_data = pd.read_csv('test.csv')

# Combine the training and test datasets for consistent feature engineering
data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

# Feature Engineering

# Extract titles from 'Name'
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Calculate family size
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# Create age groups
bins = [0, 18, 35, 50, 100]
labels = ['Child', 'Young Adult', 'Adult', 'Senior']
data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

# Create fare groups
fare_bins = [0, 15, 30, 1000]
fare_labels = ['Low', 'Medium', 'High']
data['FareGroup'] = pd.cut(data['Fare'], bins=fare_bins, labels=fare_labels, right=False)

# Create a binary 'HasCabin' feature
data['HasCabin'] = data['Cabin'].notna().astype(int)

# Encode categorical variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup'], drop_first=True)

# Drop unnecessary columns
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'SibSp', 'Parch'], axis=1, inplace=True)

# Handle missing values (fill with 0 for simplicity)
data.fillna(0, inplace=True)

# Split the data back into training and test datasets
train_data = data.iloc[:len(train_data)]
test_data = data.iloc[len(train_data):]

# Save the new datasets with new names
train_data.to_csv('train_features.csv', index=False)
test_data.to_csv('test_features.csv', index=False)
