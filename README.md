# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
# Date:1/9/2025
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Tomato dataset
file_path = "/content/Tomato.csv"
data = pd.read_csv(file_path)

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Extract Year
data['Year'] = data['Date'].dt.year

# Aggregate yearly average prices
yearly_data = data.groupby('Year')['Average'].mean().reset_index()
yearly_data.rename(columns={'Year': 'Year', 'Average': 'Price'}, inplace=True)

# Extract values
years = yearly_data['Year'].tolist()
prices = yearly_data['Price'].tolist()

# Prepare values for linear regression
X = [i - (len(years) // 2) for i in range(len(years))]
x2 = [i**2 for i in X]
xy = [i * j for i, j in zip(X, prices)]

n = len(years)
b = (n * sum(xy) - sum(prices) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(prices) - b * sum(X)) / n
linear_trend = [a + b * Xi for Xi in X]

# Polynomial Trend Estimation (Degree 2)
x3 = [i**3 for i in X]
x4 = [i**4 for i in X]
x2y = [i * j for i, j in zip(x2, prices)]

coeff = [[n, sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]

Y = [sum(prices), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)

solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * Xi + c_poly * (Xi**2) for Xi in X]

# Display trend equations
print(f"Linear Trend: y = {a:.2f} + {b:.2f}x")
print(f"Polynomial Trend: y = {a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

# Plot results
plt.figure(figsize=(12,6))
plt.plot(years, prices, 'bo-', label="Actual Avg Price")
plt.plot(years, linear_trend, 'k--', label="Linear Trend")
plt.plot(years, poly_trend, 'r-', label="Polynomial Trend")
plt.title("Tomato Price Trend Estimation")
plt.xlabel("Year")
plt.ylabel("Average Price")
plt.legend()
plt.grid(True)
plt.show()
```

### OUTPUT
<img width="1370" height="754" alt="image" src="https://github.com/user-attachments/assets/bd47a593-d0e2-49d1-b1a8-53f3b0a6de5c" />


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
