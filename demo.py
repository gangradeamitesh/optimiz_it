import numpy as np
from optimiz.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate a random regression problem
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression(learning_rate=0.01, iterations=1000, tolerance=1e-6)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate and print the R^2 score
# r2_score = model.score(X_test, y_test)
# print(f"R^2 Score: {r2_score}")

# Print the learned parameters
print(f"Intercept: {model.weights[0]}")
print(f"Coefficient: {model.weights[1]}")

# Optional: Plot the results
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Results')
plt.show()