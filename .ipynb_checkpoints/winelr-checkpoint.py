import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

wine_data = pd.read_csv('winquality.csv')

X = wine_data[['alcohol', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates']]
y = wine_data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²) Score: {r2}")


new_data = pd.DataFrame({
    'alcohol': [12.5],
    'fixed acidity': [7.2],
    'volatile acidity': [0.32],
    'citric acid': [0.36],
    'residual sugar': [6.2],
    'chlorides': [0.05],
    'free sulfur dioxide': [15],
    'total sulfur dioxide': [85],
    'density': [0.997],
    'pH': [3.3],
    'sulphates': [0.6]
})

new_prediction = model.predict(new_data)
print(f"Predicted Wine Quality for New Data: {new_prediction[0]}")
