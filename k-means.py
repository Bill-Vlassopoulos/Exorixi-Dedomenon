import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
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
    whole_data = whole_data.iloc[::100, :]
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(whole_data.drop("label", axis=1))

    return scaled_data, whole_data["label"]


def apply_kmeans(data, n_clusters=6):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters


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


def plot_elbow_method(data, max_clusters=15):
    inertia = []
    for n in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    plt.plot(range(1, max_clusters + 1), inertia, marker="o")
    plt.title("Elbow Method for Optimal Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.show()


def main():
    data, labels = load_and_preprocess_data()

    # Plot the elbow method to find the optimal number of clusters
    # plot_elbow_method(data, max_clusters=15)

    # Apply K-Means clustering with the chosen number of clusters
    kmeans_clusters = apply_kmeans(data, n_clusters=6)
    visualize_clusters(data, kmeans_clusters, "K-Means Clustering")

    # Apply DBSCAN clustering
    # dbscan_clusters = apply_dbscan(data, eps=0.5, min_samples=5)
    # visualize_clusters(data, dbscan_clusters, "DBSCAN Clustering")

    # Compare the results
    print("K-Means Clustering Results:")
    print(pd.Series(kmeans_clusters).value_counts())
    # print("\nDBSCAN Clustering Results:")
    # print(pd.Series(dbscan_clusters).value_counts())


if __name__ == "__main__":
    main()
