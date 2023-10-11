import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

wine_data = pd.read_csv('winequality.csv')

X = wine_data[['alcohol', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates']]
y = wine_data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_ftrain=X_train.fillna(0)
y_ftrain=y_train.fillna(0)

model = LinearRegression()

model.fit(X_ftrain, y_ftrain)


y_pred = model.predict(X_test.fillna(0))


mse = mean_squared_error(y_test.fillna(0), y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²) Score: {r2}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Wine Quality")
plt.ylabel("Predicted Wine Quality")
plt.title("Actual vs. Predicted Wine Quality")
plt.show()

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
