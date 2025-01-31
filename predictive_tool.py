import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset (Example: House Prices)
data = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv")

# Display first few rows
print("Dataset Sample:")
print(data.head())

# Handling missing values
data.fillna(data.mean(), inplace=True)

# Selecting Features & Target Variable
X = data[["median_income"]]  # Feature
y = data["median_house_value"]  # Target

# Split Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

# Visualization
plt.figure(figsize=(8,5))
sns.scatterplot(x=X_test["median_income"], y=y_test, label="Actual", color="blue")
sns.scatterplot(x=X_test["median_income"], y=y_pred, label="Predicted", color="red")
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("Predicting House Prices")
plt.legend()
plt.show()
