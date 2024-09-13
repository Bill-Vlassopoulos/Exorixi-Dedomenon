import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def load_and_preprocess_data():
    print("Loading data from CSV files...")
    csv_files = glob.glob("harth/*.csv")
    dataframes = {}
    for file in csv_files:
        df = pd.read_csv(file)
        key = file.split("\\")[-1].split(".")[0]
        dataframes[key] = df

    whole_data = pd.concat(dataframes.values())
    whole_data.drop(["timestamp", "index", "Unnamed: 0"], axis=1, inplace=True)
    print("Data loaded and combined successfully.")
    print(whole_data.head())
    whole_data = whole_data.iloc[::100, :]  # Sample data to reduce size for clustering

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(whole_data.drop("label", axis=1))

    return scaled_data, whole_data["label"]


def apply_agglomerative_clustering(data, n_clusters=11):
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = agglomerative.fit_predict(data)
    return clusters


def visualize_clusters(data, clusters, title):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(data)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap="viridis")
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.colorbar()
    plt.show()


def main():
    data, labels = load_and_preprocess_data()

    # Apply Agglomerative Clustering
    agglomerative_clusters = apply_agglomerative_clustering(data, n_clusters=11)
    visualize_clusters(data, agglomerative_clusters, "Agglomerative Clustering")

    # Print the Agglomerative Clustering results
    print("Agglomerative Clustering Results:")
    print(pd.Series(agglomerative_clusters).value_counts())


if __name__ == "__main__":
    main()
