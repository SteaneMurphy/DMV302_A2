from algorithms.kmeans_core import clustering_k_means, calculate_inertia, calculate_dunn_index

"""
K-Means clustering experiement (K2 - K10).
Calculates inertia and dunn index and prints results.
"""
inertia_result = []
dunn_result = []

for i in range(2, 11):
    centroids, clusters = clustering_k_means(
        k_value=i, 
        max_iterations=100, 
        tolerance=0.001
    )

    inertia = calculate_inertia(centroids, clusters)
    dunn_index = calculate_dunn_index(centroids, clusters) 

    inertia_result.append(inertia)
    dunn_result.append(dunn_index)

"""
RESULTS
"""
print("\n K   |   Inertia     |   Dunn Index")
print("-------------------------------------")

k_values = list(range(2, 11))

for i in range(len(k_values)):
    k = k_values[i]
    inertia = inertia_result[i] / 1000
    dunn = dunn_result[i]

    print(f"{k:>3}  | {inertia:>10.4f}    | {dunn:>10.4f}")