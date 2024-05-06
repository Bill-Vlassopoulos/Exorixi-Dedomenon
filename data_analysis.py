import pandas as pd
import matplotlib.pyplot as plt
import glob


def main():
    csv_files = glob.glob("harth/*.csv")
    dataframes = {}
    for file in csv_files:
        df = pd.read_csv(file)
        key = file.split("\\")[-1].split(".")[0]  # Get file name without extension
        dataframes[key] = df
    print(dataframes.keys())
    # print(data6.head(1))
    # data6["timestamp"] = pd.to_datetime(data6["timestamp"])
    # mean_values = data6.drop(columns=["timestamp"]).groupby("label").mean()
    # print(mean_values)


if __name__ == "__main__":

    main()
