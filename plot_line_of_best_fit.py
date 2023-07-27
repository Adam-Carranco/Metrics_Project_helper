import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr

def plot_polynomial_best_fit(df):
    # Convert the "Pass/Fail Status.1" column to a binary format (1 for pass, 0 for fail)
    df["Pass/Fail Status.1"] = df["Pass/Fail Status.1"].replace({3: 1, 2: None, 1: 0})

    # Remove rows with None values in the "Pass/Fail Status.1" column
    df = df.dropna(subset=["Pass/Fail Status.1"])

    # Iterate through each column (excluding "Pass/Fail Status.1") and plot it against the target column
    for column in df.columns[1:]:
        # Prepare the input features and target variable
        X = df[[column]]
        y = df["Pass/Fail Status.1"]

        # Create third-degree polynomial features
        polynomial_features = PolynomialFeatures(degree=3)
        X_poly = polynomial_features.fit_transform(X)

        # Fit a polynomial regression model
        polynomial_model = LinearRegression()
        polynomial_model.fit(X_poly, y)

        # Calculate the correlation coefficient (R-squared) between predicted and actual values
        y_pred = polynomial_model.predict(X_poly)
        r_squared = pearsonr(y, y_pred)[0] ** 2

        # Plot the data and the third-degree polynomial regression line
        plt.scatter(X, y, color="b", label="Data")

        # Sort the data for smoother plot
        X_sorted, y_pred_sorted = zip(*sorted(zip(X.values, y_pred)))

        plt.plot(X_sorted, y_pred_sorted, color="r", label=f"3rd Degree Polynomial (R-squared: {r_squared:.2f})")

        plt.xlabel(column)
        plt.ylabel("Pass/Fail Status")
        plt.title(f"{column} vs. Pass/Fail Status")
        plt.legend()
        plt.show()

# Example usage:
# Assuming your DataFrame is named "data_frame"
plot_polynomial_best_fit(data_frame)
