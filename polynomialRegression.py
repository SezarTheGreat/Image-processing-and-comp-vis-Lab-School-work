import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 20)
y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-3, 3, 20)

# Reshape for sklearn
X = X[:, np.newaxis]
y = y[:, np.newaxis]

# --- Polynomial Regression (Degree 3) ---
degree = 3
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# Sort X for plotting the curve smoothly
sort_axis = np.argsort(X[:, 0])
X_sorted = X[sort_axis]
y_pred_poly_sorted = y_pred_poly[sort_axis]

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y, y_pred_poly))
r2 = r2_score(y, y_pred_poly)

print(f"Polynomial Regression (Degree {degree}) Results:")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Predict for a user input
user_input = float(input("\nEnter a value for X to predict y: "))
user_input_poly = poly_features.transform([[user_input]])
user_prediction = poly_reg.predict(user_input_poly)
print(f"Predicted y for X={user_input}: {user_prediction[0][0]:.2f}")

# --- Plotting ---
plt.scatter(X, y, s=10, color='blue', label='Data points')
plt.plot(X_sorted, y_pred_poly_sorted, color='green', label=f'Polynomial Regression (Degree {degree})')
plt.scatter(user_input, user_prediction, color='red', s=100, marker='x', label=f'Predicted: {user_prediction[0][0]:.2f}')

plt.title('Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()