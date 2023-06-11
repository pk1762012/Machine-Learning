import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the Titanic dataset
titanic_df = pd.read_csv('titanic.csv')

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

# Calculate the within-cluster sum of squares (WCSS) for different numbers of clusters
wcss = []
silhouette_scores = []
max_clusters = 10

for n_clusters in range(2, max_clusters+1):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(2, max_clusters+1), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Curve')
plt.show()

# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, max_clusters+1), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores')
plt.show()
