import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr

def plot_best_fit(x_column, y_column, degree=1):
    # Assuming your DataFrame is named "data_frame"
    X = data_frame[[x_column]]
    y = data_frame[y_column]

    if degree > 1:
        # Create polynomial features
        polynomial_features = PolynomialFeatures(degree=degree)
        X_poly = polynomial_features.fit_transform(X)

        # Fit a polynomial regression model
        polynomial_model = LinearRegression()
        polynomial_model.fit(X_poly, y)

        # Calculate the correlation coefficient (R-squared) between predicted and actual values
        y_pred = polynomial_model.predict(X_poly)
        r_squared = pearsonr(y, y_pred)[0] ** 2

        # Sort the data for smoother plot
        X_sorted, y_pred_sorted = zip(*sorted(zip(X.values, y_pred)))

        plt.plot(X_sorted, y_pred_sorted, color="r", label=f"{degree} Degree Polynomial (R-squared: {r_squared:.2f})")
    else:
        # Fit a linear regression model
        linear_model = LinearRegression()
        linear_model.fit(X, y)

        # Calculate the correlation coefficient (R-squared) between predicted and actual values
        y_pred = linear_model.predict(X)
        r_squared = pearsonr(y, y_pred)[0] ** 2

        # Sort the data for smoother plot
        X_sorted, y_pred_sorted = zip(*sorted(zip(X.values, y_pred)))

        plt.plot(X_sorted, y_pred_sorted, color="r", label=f"Linear Fit (R-squared: {r_squared:.2f})")

    plt.scatter(X, y, color="b", label="Data")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f"{x_column} vs. {y_column}")
    plt.legend()
    plt.show()

# Example usage:
# Assuming your DataFrame is named "data_frame"
# Replace "your_x_column" and "your_y_column" with the desired column names from your DataFrame
plot_best_fit("your_x_column", "your_y_column", degree=3)
