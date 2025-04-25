import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load data with warning fix
data = pd.read_csv(r'C:\Users\Win10\Desktop\muruga\Housing.csv',low_memory=False)

# Fill NaNs with column means
data_filled = data.fillna(data.mean(numeric_only=True))

# Select only numeric columns
numeric_data = data_filled.select_dtypes(include=['number'])

# Scale the numeric data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Apply KMeans with k=3
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(scaled_data)

# Inertia and Silhouette Score
inertia = kmeans.inertia_
sil_score = silhouette_score(scaled_data, labels)
print(f"KMeans clustering completed.\nInertia: {inertia}\nSilhouette Score: {sil_score:.4f}")

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
plt.title('Cluster Visualization using PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()