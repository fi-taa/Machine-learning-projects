# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load your dataset
data = pd.read_csv('StudentsPerformance.csv')  

# Define your feature columns (X) and target variables (y)
X = data[['gender', 'parental level of education', 'lunch', 'test preparation course']]
y = data[['math score', 'reading score', 'writing score']]

# Encode categorical features if needed (convert strings to numeric values)
X_encoded = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize the multi-output regression model (e.g., Linear Regression)
model = MultiOutputRegressor(LinearRegression())

# Train the model on the training data
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'multioutput_regression_model.pkl')

# Evaluate the model's performance (you can use different metrics for each target)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared Score: {r2}')
