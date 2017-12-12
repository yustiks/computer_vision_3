from sklearn.cluster import KMeans
import numpy as np
f1 = [2,17,1,11,23]
f2 = [-3,15,7,14,2]
x = np.array(list(zip(f1, f2)))
# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(x)
# Getting the cluster labels
labels = kmeans.predict(x)
# Centroid values
centroids = kmeans.cluster_centers_
print (centroids)