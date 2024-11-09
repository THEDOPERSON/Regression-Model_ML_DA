# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv(r'C:\Users\ASUS\Downloads\ML_Q1DA\used_cars.csv')  # Adjust the path if needed

# Data Cleaning
# Remove non-numeric characters from 'milage' and 'price', then convert to numeric
data['milage'] = data['milage'].str.replace(r'[^0-9]', '', regex=True).astype(float)
data['price'] = data['price'].str.replace(r'[^0-9]', '', regex=True).astype(float)

# Select numeric and relevant columns for regression
data = data[['model_year', 'milage', 'price']]

# Drop rows with missing values (if any)
data.dropna(inplace=True)

# Define features (X) and target (y)
X = data[['model_year', 'milage']]
y = data['price']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the results
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
