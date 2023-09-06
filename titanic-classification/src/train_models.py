import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import joblib

# Load your preprocessed dataset
# Replace 'your_dataset.csv' with your actual dataset file path
combined_data = pd.read_csv('train_features.csv')  # Assuming 'train_features' contains both features and labels

# Split the combined data into train and test sets
train_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42)

# Separate features and labels
train_labels = train_data['Survived']
test_labels = test_data['Survived']

# Extract features by dropping the 'Survived' column
train_features = train_data.drop('Survived', axis=1)
test_features = test_data.drop('Survived', axis=1)

# Define functions for training each model
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_support_vector_machine(X_train, y_train):
    model = SVC(kernel='linear', C=1.0, probability=True)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = XGBClassifier(learning_rate=0.1, n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_neural_network(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

# Train each model and save them to separate files
logistic_regression_model = train_logistic_regression(train_features, train_labels)
joblib.dump(logistic_regression_model, 'logistic_regression_model.pkl')

random_forest_model = train_random_forest(train_features, train_labels)
joblib.dump(random_forest_model, 'random_forest_model.pkl')

svm_model = train_support_vector_machine(train_features, train_labels)
joblib.dump(svm_model, 'svm_model.pkl')

xgboost_model = train_xgboost(train_features, train_labels)
joblib.dump(xgboost_model, 'xgboost_model.pkl')

neural_network_model = train_neural_network(train_features, train_labels)
joblib.dump(neural_network_model, 'neural_network_model.pkl')
