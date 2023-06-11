import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pickle
import os

# Create folders if they don't exist
output_folder = 'files_saved'
pickle_folder = 'pickle_files'
# Load the Titanic dataset
titanic_file = os.path.join(output_folder, 'titanic.csv')
titanic_df = pd.read_csv(titanic_file)

# Select relevant features for clustering
selected_features = ['Age', 'Fare', 'Pclass', 'Sex']

# Preprocess the data
X = titanic_df[selected_features].values

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X[:, :-1])  # Exclude the last column (Sex)

# Perform one-hot encoding for the 'Sex' feature
onehot_encoder = OneHotEncoder(sparse=False)
sex_encoded = onehot_encoder.fit_transform(X[:, -1].reshape(-1, 1))

# Concatenate the encoded features with the remaining numerical features
X_combined = np.concatenate((X_imputed, sex_encoded), axis=1)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Perform clustering with different algorithms
kmeans = KMeans(n_clusters=5)
agglomerative = AgglomerativeClustering(n_clusters=5)
dbscan = DBSCAN(eps=0.5, min_samples=5)

algorithms = {
    'KMeans': kmeans,
    'Agglomerative': agglomerative,
    'DBSCAN': dbscan
}

best_algorithm = None
best_silhouette_score = -1

# Iterate over the algorithms
for name, algorithm in algorithms.items():
    algorithm.fit(X_scaled)
    labels = algorithm.labels_
    silhouette = silhouette_score(X_scaled, labels)
    print(f'{name} Silhouette Score: {silhouette}')

    if silhouette > best_silhouette_score:
        best_algorithm = algorithm
        best_silhouette_score = silhouette

print(f'Best Clustering Algorithm: {best_algorithm}')

# Inverse transform the scaled data
X_original = scaler.inverse_transform(X_scaled)
# Reverse the one-hot encoding for the 'Sex' feature
sex_decoded = onehot_encoder.inverse_transform(X_original[:, -2:])[:, 0]  # Select the last two columns (encoded 'Sex')

# Concatenate the non-encoded 'Sex' feature with the remaining features
X_decoded = np.concatenate((X_original[:, :-2], sex_decoded.reshape(-1, 1)), axis=1)


# Get the cluster labels assigned by the best algorithm
labels = best_algorithm.labels_

# Get the unique cluster labels
unique_labels = np.unique(labels)

# Iterate through the unique cluster labels
for cluster_label in unique_labels:
    # Filter the data points belonging to the current cluster
    cluster_data = X_decoded[labels == cluster_label]
    # Print or perform further analysis on the cluster data
    print(f"Cluster {cluster_label}:")
    print(cluster_data)
    print()



# Visualize the clustering results
plt.scatter(X_original[:, 0], X_original[:, 1], c=best_algorithm.labels_, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustering Results')
plt.savefig(os.path.join(output_folder, 'clustering_results.png'))
plt.show()


# Save the best algorithm as a pickle file
pickle_file = os.path.join(pickle_folder, 'best_clustering.pkl')
# with open(pickle_file, 'wb') as f:
#     pickle.dump(best_algorithm, f)
