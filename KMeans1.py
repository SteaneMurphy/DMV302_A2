from algorithms.kmeans_core import clustering_k_means

"""
KMeans clustering experiment (K=3).
Runs clustering on standardised financial data and prints centroids.
"""
centroids, clusters = clustering_k_means(
    k_value=4, 
    max_iterations=100, 
    tolerance=0.001
)

"""
RESULTS
"""
print("\n           Assets    |     Income")
print("-------------------------------------")
for i, c in enumerate(centroids, start=1):
    print(f"Centroid {i:<2}: {c[0]:>10.6f}   {c[1]:>10.6f}")