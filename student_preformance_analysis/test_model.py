# test_model.py
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('multioutput_regression_model.pkl')

# Sample user data (you can modify this data as needed)
sample_data = {
    'gender_male': [1],  # 1 for Male, 0 for Female
    'parental level of education_some high school': [1],  # 1 for 'some high school', 0 for others
    'lunch_standard': [1],  # 1 for Standard, 0 for Free/Reduced
    'test preparation course_none': [1]  # 1 for None, 0 for Completed
}

# Create a DataFrame with sample data
user_data = pd.DataFrame(sample_data)

# Make predictions
predictions = model.predict(user_data)

# Display predictions
print('Predicted Scores:')
print(f'Math Score: {predictions[0, 0]}')
print(f'Reading Score: {predictions[0, 1]}')
print(f'Writing Score: {predictions[0, 2]}')
