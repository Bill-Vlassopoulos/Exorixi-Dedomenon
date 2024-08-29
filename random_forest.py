import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def main():
    print("Hello from supervised_learning.py")
    csv_files = glob.glob("harth/*.csv")
    dataframes = {}
    for file in csv_files:
        df = pd.read_csv(file)
        key = file.split("\\")[-1].split(".")[0]  # Get file name without extension
        dataframes[key] = df

    whole_data = pd.concat(dataframes.values())
    whole_data = whole_data.iloc[::5].reset_index(drop=True)

    Χ = whole_data.drop(["timestamp", "label"], axis=1)
    y = whole_data["label"]

    Χ_train, Χ_test, y_train, y_test = train_test_split(
        Χ, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features="log2",
        max_depth=None,
        bootstrap=False,
        n_jobs=-1,
    )

    model.fit(Χ_train, y_train)

    y_pred = model.predict(Χ_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
