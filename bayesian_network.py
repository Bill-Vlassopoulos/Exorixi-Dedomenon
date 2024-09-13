import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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
    whole_data.drop(["index", "Unnamed: 0"], axis=1, inplace=True)
    print(whole_data.head())

    # Split the data into training and testing sets
    X = whole_data.drop(["timestamp", "label"], axis=1)
    y = whole_data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define and fit the Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = model.predict(X_test)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Print the classification report
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
