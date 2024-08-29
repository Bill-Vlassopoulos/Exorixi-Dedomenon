import pandas as pd
import glob
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def main():
    print("Hello from bayesian_network.py")
    csv_files = glob.glob("harth/*.csv")
    dataframes = {}
    for file in csv_files:
        df = pd.read_csv(file)
        key = file.split("\\")[-1].split(".")[0]  # Get file name without extension
        dataframes[key] = df

    whole_data = pd.concat(dataframes.values())
    print(whole_data.head())
    # Split the data into training and testing sets
    X = whole_data.drop(["timestamp", "label"], axis=1)
    y = whole_data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define the Bayesian Network structure
    model = BayesianNetwork(
        [
            ("back_x", "label"),
            ("back_y", "label"),
            ("back_z", "label"),
            ("thigh_x", "label"),
            ("thigh_y", "label"),
            ("thigh_z", "label"),
        ]
    )

    # Fit the model
    model.fit(X_train, estimator=MaximumLikelihoodEstimator)

    # Predict the labels for the test set
    y_pred = model.predict(X_test)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Print the classification report
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
