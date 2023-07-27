
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

def find_best_predictor(df):
    # Extract columns containing data metrics (excluding "pass/fail status")
    data_columns = df.columns[1:]

    # Initialize variables to store the best correlation and corresponding features
    best_correlation = -1
    best_features = []

    # Iterate through all possible combinations of metrics with at most 4 metrics at a time
    max_metrics_combinations = 4
    for r in range(1, min(max_metrics_combinations, len(data_columns)) + 1):
        for combination in itertools.combinations(data_columns, r):
            # Combine the selected columns with the "pass/fail status" column
            selected_columns = ["pass/fail status"] + list(combination)
            selected_df = df[selected_columns]

            # Convert the "pass/fail status" column to a binary format (1 for pass, 0 for fail)
            selected_df["pass/fail status"] = selected_df["pass/fail status"].replace({3: 1, 2: 1, 1: 0})

            # Prepare the input features and target variable
            X = selected_df.iloc[:, 1:]
            y = selected_df.iloc[:, 0]

            # Fit a linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Calculate the correlation coefficient between predicted and actual values
            y_pred = model.predict(X)
            correlation, _ = pearsonr(y, y_pred)

            # Update the best correlation and features if necessary
            if correlation > best_correlation:
                best_correlation = correlation
                best_features = selected_columns[1:]

    # Re-create the selected DataFrame with the best combination of features
    best_df = df[["pass/fail status"] + best_features]

    # Plot the regression lines for each feature
    num_plots = len(best_features)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 6*num_plots), sharex=True)
    fig.suptitle("Linear Regression - Best Predictors", fontsize=16)

    for i, feature in enumerate(best_features):
        ax = axes[i]
        ax.scatter(best_df[feature], best_df["pass/fail status"], color="b", label="Data")
        ax.plot(best_df[feature], model.predict(best_df[[feature]]), color="r", label="Regression Line")
        ax.set_ylabel("Pass/Fail Status")
        ax.set_title(f"{feature} vs. Pass/Fail Status")
        ax.legend()

    plt.xlabel("Metrics")
    plt.show()

    return best_df, best_features, best_correlation

# Example usage:
# Assuming your DataFrame is named "data_frame"
best_df, best_features, best_correlation = find_best_predictor(data_frame)
print("Best Features:", best_features)
print("Best Correlation:", best_correlation)

