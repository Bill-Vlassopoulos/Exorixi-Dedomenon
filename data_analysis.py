import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import numpy as np
from mpl_toolkits import mplot3d
import seaborn as sns


def main():
    csv_files = glob.glob("harth/*.csv")
    dataframes = {}
    for file in csv_files:
        df = pd.read_csv(file)
        key = file.split("\\")[-1].split(".")[0]  # Get file name without extension
        dataframes[key] = df

    whole_data = pd.concat(dataframes.values()).drop(["index", "Unnamed: 0"], axis=1)
    whole_data_grouped = whole_data.groupby("label")
    # sensor_stats(whole_data_grouped)

    # activities_per_user(dataframes["S006"], "S006")

    plot_boxplots(whole_data)
    plot_correlation_heatmap(whole_data.drop(["timestamp"], axis=1))


def sensor_stats(data_grouped):
    for activity, group in data_grouped:
        # Υπολογισμός μέσου όρου και τυπικής απόκλισης για κάθε στήλη εκτός από 'timestamp' και 'label'
        stats = group.agg(
            {
                col: ["mean", "std"]
                for col in group.columns
                if col not in ["timestamp", "label", "index", "Unnamed: 0"]
            }
        )
        print(f"\nActivity: {activity}")
        print(stats)


def get_correlations(data):
    columns = list(data.columns)[1:-3]
    scores = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            correlation = data[columns[i]].corr(data[columns[j]])
            scores.append(
                [
                    f"{columns[i]} - {columns[j]}:",
                    correlation.__round__(4),
                ]
            )
    scores.sort(key=lambda x: x[1], reverse=True)
    print("\n\n Correlation Scores:")
    for score in scores:
        print(score[0], score[1])


def activities_per_user(user_dataframe, user_id):
    user_dataframe["group"] = (
        user_dataframe["label"] != user_dataframe["label"].shift()
    ).cumsum()
    grouped = user_dataframe.groupby("group")
    # first and last index of each group
    print(
        f"""
          ------------------------- {user_id} --------------------------
          """
    )
    first_index = grouped.head(1).index
    last_index = grouped.tail(1).index
    activities_list = []
    duration_list = []
    next_activity = {}
    all_activities = user_dataframe["label"].unique()
    all_activities = list(all_activities)
    for i in range(len(first_index)):
        activity = user_dataframe.iloc[first_index[i]]["label"]
        start = user_dataframe.iloc[first_index[i]]["timestamp"]
        end = user_dataframe.iloc[last_index[i]]["timestamp"]
        duration = pd.to_datetime(end) - pd.to_datetime(start)
        if activity not in next_activity:
            next_activity[activity] = [0 for _ in range(len(all_activities))]
        if last_index[i] < last_index[-1]:
            idx = all_activities.index(user_dataframe.iloc[last_index[i] + 1]["label"])
            next_activity[activity][idx] += 1
        print(
            f"Activity: {activity}, Start: {start}, End: {end}, Duration: {duration.total_seconds():.2f}s"
        )
        if activity not in activities_list:
            activities_list.append(activity)
            duration_list.append(duration.total_seconds())
        else:
            index = activities_list.index(activity)
            duration_list[index] += duration.total_seconds()

    sorted_indices = np.argsort(duration_list)[::-1]
    activities_list = [activities_list[i] for i in sorted_indices]
    duration_list = [duration_list[i] for i in sorted_indices]
    print(
        """
------------ Overall Activities ------------
    """
    )
    first_labels = grouped.first()["label"]
    for i in range(len(activities_list)):
        print(
            f"Activity: {activities_list[i]}, Duration: {duration_list[i]:.2f}s, Mean Lap Duration: {(duration_list[i]/(first_labels == activities_list[i]).sum()):.2f}s, Most Likely Next Activity: {all_activities[np.argmax(next_activity[activities_list[i]])]}({max(next_activity[activities_list[i]])}/{sum(next_activity[activities_list[i]])})"
        )
    print(
        """
-------------------------------------------"""
    )
    overall_duration(user_dataframe)


def overall_duration(user_dataframe):
    first_index = user_dataframe.head(1).index
    last_index = user_dataframe.tail(1).index
    start = user_dataframe.iloc[first_index[0]]["timestamp"]
    end = user_dataframe.iloc[last_index[-1]]["timestamp"]
    duration = pd.to_datetime(end) - pd.to_datetime(start)

    print(f"Overall Duration: {duration.total_seconds()/60:.2f} mins")


def plot_boxplots(whole_data):
    columns = [col for col in whole_data.columns if col not in ["timestamp", "label"]]
    for column in columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="label", y=column, data=whole_data)
        plt.title(f"Boxplot για την {column} ανά δραστηριότητα")
        plt.show()


# Heatmap συσχετίσεων
def plot_correlation_heatmap(data):

    corr_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.show()


if __name__ == "__main__":
    main()
