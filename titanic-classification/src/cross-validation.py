import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, f1_score
import joblib

# Load your dataset (train_features should contain both features and labels)
train_features = pd.read_csv('train_features.csv')
test_features = pd.read_csv('test_features.csv')  # Load the test dataset without 'Survived'

# Separate features and labels
X_train = train_features.drop('Survived', axis=1)
y_train = train_features['Survived']



# Remove 'Survived' column from the test features
test_features = test_features.drop('Survived', axis=1)

# Models to evaluate (assuming you have trained and saved these models)
models = {
    'Logistic Regression': joblib.load('logistic_regression_model.pkl'),
    'Random Forest': joblib.load('random_forest_model.pkl'),
    'Support Vector Machine': joblib.load('svm_model.pkl'),
    'XGBoost': joblib.load('xgboost_model.pkl'),
    'Neural Network': joblib.load('neural_network_model.pkl'),
}

# Cross-validation and evaluation on the test set
for model_name, model in models.items():
    print(f'Evaluating {model_name}:')

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f'Cross-Validation Scores: {cv_scores}')
    print(f'Mean CV Score: {cv_scores.mean()}')

    # Make predictions on the test set
    y_pred = model.predict(test_features)

    # Save the test predictions to a file if needed
    pd.DataFrame({ 'Survived': y_pred}).to_csv(f'{model_name}_test_predictions.csv', index=False)

    print(f'{model_name} evaluation completed.\n')

print('All model evaluations completed.')
