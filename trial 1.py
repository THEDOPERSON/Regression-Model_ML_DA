
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv(r'C:\Users\ASUS\Downloads\ML_Q1DA\used_cars.csv')  # Adjust the path if needed


data['milage'] = data['milage'].str.replace(r'[^0-9]', '', regex=True).astype(float)
data['price'] = data['price'].str.replace(r'[^0-9]', '', regex=True).astype(float)


data = data[['model_year', 'milage', 'price']]


data.dropna(inplace=True)


X = data[['model_year', 'milage']]
y = data['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
