import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def main():
    print("Hello from supervised_learning.py")
    csv_files = glob.glob("harth/*.csv")
    dataframes = {}
    for file in csv_files:
        df = pd.read_csv(file)
        key = file.split("\\")[-1].split(".")[0]  # Get file name without extension
        dataframes[key] = df

    whole_data = pd.concat(dataframes.values())
    whole_data = whole_data.iloc[::50].reset_index(drop=True)

    Χ = whole_data.drop(["timestamp", "label"], axis=1)
    y = whole_data["label"]

    Χ_train, Χ_test, y_train, y_test = train_test_split(
        Χ, y, test_size=0.2, random_state=42
    )

    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint

    param_dist = {
        "n_estimators": [50, 100],  # Reduced number of estimators
        "max_features": ["sqrt", None],  # Reduced max_features options
        "max_depth": [10, 20],  # Set a max depth limit
        "min_samples_split": [5, 10],
        "min_samples_leaf": [1, 2],
        "bootstrap": [True],  # Only test with bootstrap enabled
    }

    model = RandomForestClassifier(random_state=42, n_jobs=-1)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )

    random_search.fit(Χ_train, y_train)

    best_params = random_search.best_params_
    print(f"Best parameters: {best_params}")

    # # Train the model with the best parameters
    # best_model = random_search.best_estimator_
    # best_model.fit(Χ_train, y_train)
    # y_pred = best_model.predict(Χ_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Optimized Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
