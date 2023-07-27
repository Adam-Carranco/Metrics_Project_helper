import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr

def find_best_predictor(df):
    # Extract columns containing data metrics (excluding "Pass/Fail Status.1")
    data_columns = df.columns[1:]

    # Initialize variables to store the best correlation and corresponding features
    best_correlation = -1
    best_features = []

    # Maximum number of metrics to compare
    max_metrics_combinations = 10
    num_metrics = min(max_metrics_combinations, len(data_columns))

    # Iterate through all possible combinations of metrics with at most 10 metrics at a time
    for r in range(1, num_metrics + 1):
        for combination in itertools.combinations(data_columns, r):
            # Combine the selected columns with the "Pass/Fail Status.1" column
            selected_columns = ["Pass/Fail Status.1"] + list(combination)
            selected_df = df[selected_columns]

            # Convert the "Pass/Fail Status.1" column to a binary format (1 for pass, 0 for fail)
            selected_df["Pass/Fail Status.1"] = selected_df["Pass/Fail Status.1"].replace({3: 1, 2: 1, 1: 0})

            # Prepare the input features and target variable
            X = selected_df.iloc[:, 1:]
            y = selected_df.iloc[:, 0]

            # Fit a polynomial regression model
            polynomial_features = PolynomialFeatures(degree=2)  # You can set the degree as desired (e.g., 2, 3, 4, ...)
            X_poly = polynomial_features.fit_transform(X)

            model = LinearRegression()
            model.fit(X_poly, y)

            # Calculate the correlation coefficient between predicted and actual values
            y_pred = model.predict(X_poly)
            correlation, _ = pearsonr(y, y_pred)

            # Update the best correlation and features if necessary
            if correlation > best_correlation:
                best_correlation = correlation
                best_features = selected_columns[1:]

    # Re-create the selected DataFrame with the best combination of features
    best_df = df[["Pass/Fail Status.1"] + best_features]

    # Plot the regression lines for each feature
    num_plots = len(best_features)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 6*num_plots), sharex=True)
    fig.suptitle("Polynomial Regression - Best Predictors", fontsize=16)

    for i, feature in enumerate(best_features):
        ax = axes[i]
        sorted_indices = best_df[feature].argsort()
        ax.scatter(best_df[feature], best_df["Pass/Fail Status.1"], color="b", label="Data")

        # Create polynomial features for plotting the regression line
        X_plot = np.linspace(min(best_df[feature]), max(best_df[feature]), 100).reshape(-1, 1)
        X_plot_poly = polynomial_features.transform(X_plot)
        y_plot = model.predict(X_plot_poly)
        ax.plot(X_plot, y_plot, color="r", label="Regression Line")

        ax.set_ylabel("Pass/Fail Status")
        ax.set_title(f"{feature} vs. Pass/Fail Status")
        ax.legend()

        # Set individual axis limits for each subplot
        ax.set_xlim(min(best_df[feature]), max(best_df[feature]))
        ax.set_ylim(min(best_df["Pass/Fail Status.1"]), max(best_df["Pass/Fail Status.1"]))

    plt.xlabel("Metrics")
    plt.show()

    return best_df, best_features, best_correlation

# Example usage:
# Assuming your DataFrame is named "data_frame"
best_df, best_features, best_correlation = find_best_predictor(data_frame)
print("Best Features:", best_features)
print("Best Correlation:", best_correlation)
