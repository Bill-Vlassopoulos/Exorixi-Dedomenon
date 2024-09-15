import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
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


def apply_dbscan(data, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data)
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
    # Apply DBSCAN clustering
    dbscan_clusters = apply_dbscan(data, eps=1.5, min_samples=10)
    visualize_clusters(data, dbscan_clusters, "DBSCAN Clustering")

    # Calculate and print the silhouette score
    if len(set(dbscan_clusters)) > 1:  # Silhouette score requires at least 2 clusters
        score = silhouette_score(data, dbscan_clusters)
        print(f"Silhouette Score: {score}")
    else:
        print("Silhouette Score cannot be calculated with less than 2 clusters.")

    # Print the DBSCAN Clustering results
    print("DBSCAN Clustering Results:")
    print(pd.Series(dbscan_clusters).value_counts())


if __name__ == "__main__":
    main()
