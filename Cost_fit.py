import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

def plot_linear_best_fit(df):
    # Convert the "Pass/Fail Status.1" column to a binary format (1 for pass, 0 for fail)
    df["Pass/Fail Status.1"] = df["Pass/Fail Status.1"].replace({3: 1, 2: None, 1: 0})

    # Remove rows with None values in the "Pass/Fail Status.1" column
    df = df.dropna(subset=["Pass/Fail Status.1"])

    # Prepare the input features and target variable
    y = df["Pass/Fail Status.1"]

    # Iterate through each column (excluding "Pass/Fail Status.1") and plot it against "Total Incremental Cost"
    for column in df.columns[1:]:
        if column != "Total Incremental Cost":
            X = df[[column]]

            # Fit a linear regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)

            # Calculate the correlation coefficient (R-squared) between predicted and actual values
            y_pred = linear_model.predict(X)
            r_squared = pearsonr(y, y_pred)[0] ** 2

            # Plot the data and the linear regression line
            plt.scatter(X, y, color="b", label="Data")
            plt.plot(X, y_pred, color="r", label=f"Linear Fit (R-squared: {r_squared:.2f})")

            plt.xlabel(column)
            plt.ylabel("Pass/Fail Status")
            plt.title(f"{column} vs. Pass/Fail Status")
            plt.legend()
            plt.show()

    # Plot "Total Incremental Cost" against the "Pass/Fail Status.1" metric
    X_total_inc_cost = df[["Total Incremental Cost"]]

    # Fit a linear regression model for "Total Incremental Cost" against the target variable
    linear_model_total_cost = LinearRegression()
    linear_model_total_cost.fit(X_total_inc_cost, y)

    # Calculate the correlation coefficient (R-squared) between predicted and actual values
    y_pred_total_cost = linear_model_total_cost.predict(X_total_inc_cost)
    r_squared_total_cost = pearsonr(y, y_pred_total_cost)[0] ** 2

    # Plot the data and the linear regression line
    plt.scatter(X_total_inc_cost, y, color="b", label="Data")
    plt.plot(X_total_inc_cost, y_pred_total_cost, color="r", label=f"Linear Fit (R-squared: {r_squared_total_cost:.2f})")

    plt.xlabel("Total Incremental Cost")
    plt.ylabel("Pass/Fail Status")
    plt.title("Total Incremental Cost vs. Pass/Fail Status")
    plt.legend()
    plt.show()

# Example usage:
# Assuming your DataFrame is named "data_frame"
plot_linear_best_fit(data_frame)
