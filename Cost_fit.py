import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

def plot_linear_best_fit(df):
    # Convert the "Pass/Fail Status.1" column to a binary format (1 for pass, 0 for fail)
    df["Pass/Fail Status.1"] = df["Pass/Fail Status.1"].replace({3: 1, 2: None, 1: 0})

    # Remove rows with None values in the "Pass/Fail Status.1" column
    df = df.dropna(subset=["Pass/Fail Status.1"])

    # Prepare the target variable
    y = df["Total Incremental Cost"]

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
            plt.ylabel("Total Incremental Cost")
            plt.title(f"{column} vs. Total Incremental Cost")
            plt.legend()
            plt.show()

# Plot "Total Incremental Cost" against the "Pass/Fail Status.1" metric
def plot_total_cost_vs_pass_fail(df):
    # Convert the "Pass/Fail Status.1" column to a binary format (1 for pass, 0 for fail)
    df["Pass/Fail Status.1"] = df["Pass/Fail Status.1"].replace({3: 1, 2: None, 1: 0})

    # Remove rows with None values in the "Pass/Fail Status.1" column
    df = df.dropna(subset=["Pass/Fail Status.1"])

    # Prepare the input features and target variable
    X = df[["Pass/Fail Status.1"]]
    y = df["Total Incremental Cost"]

    # Fit a linear regression model
    linear_model = LinearRegression()
    linear_model.fit(X, y)

    # Calculate the correlation coefficient (R-squared) between predicted and actual values
    y_pred = linear_model.predict(X)
    r_squared = pearsonr(y, y_pred)[0] ** 2

    # Plot the data and the linear regression line
    plt.scatter(X, y, color="b", label="Data")
    plt.plot(X, y_pred, color="r", label=f"Linear Fit (R-squared: {r_squared:.2f})")

    plt.xlabel("Pass/Fail Status")
    plt.ylabel("Total Incremental Cost")
    plt.title("Total Incremental Cost vs. Pass/Fail Status")
    plt.legend()
    plt.show()

# Example usage:
# Assuming your DataFrame is named "data_frame"
plot_linear_best_fit(data_frame)
plot_total_cost_vs_pass_fail(data_frame)
