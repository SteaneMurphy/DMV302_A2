import random
import numpy as np
from numpy import mean
from utility import read_csv, parse_numeric, standardise, euclidean_distance

def preprocess_data(path:str):
    rows = read_csv(path, skip_header=True)                                                    # import rows from CSV
    features = parse_numeric(rows)                                                             # parse rows into numeric data format
    features = standardise(features)                                                           # standardise the rows for clustering

    return features



def clustering_k_means(k_value:int, max_iterations:int, tolerance:float | None = None):
    data = preprocess_data("data/household_wealth.csv")                                        # preprocess data - standardise

    centroids = random.sample(list(data), k_value)                                             # choose centroids, k-means

    for _ in range(max_iterations):                                                            # loop for max_iterations
        clusters = [[] for _ in range(k_value)]                                                # init array for data point clusters

        for vector in data:                                                                    # for each data point in data
            distances = [euclidean_distance(vector, centroid) for centroid in centroids]       # calculate the euclid distance from data point to each centroid
            closest = distances.index(min(distances))                                          # determine closest centroid to data point
            clusters[closest].append(vector)                                                   # assign data point to closest centroid
                               
        new_centroids = [mean(cluster, axis=0) if cluster else centroids[i]                    # calculate the mean for centroids from all assigned data points
            for i, cluster in enumerate(clusters)                                              # for each cluster (enumerate over clusters array)
        ]

        shifts = [euclidean_distance(centroids[i], new_centroids[i])                           # calculate the euclid distance from old centroids to new centroids
            for i in range(k_value)                                                            # for each centroid (centroids vs new_centroids)
        ]

        if tolerance is not None and max(shifts) < tolerance:                                  # check largest centroid movement is below tolerance
            break                                                                              # end algorithm if true

        centroids = new_centroids                                                              # store reference to new centroids                                                   

    return centroids, clusters

#https://numpy.org/doc/stable/reference/generated/numpy.mean.html



def calculate_inertia(centroids:list[np.ndarray], clusters:list[list[np.ndarray]]) -> float:
    inertia = 0.0                                                                              # init inertia

    for i, cluster in enumerate(clusters):                                                     # for each cluster
        centroid = centroids[i]                                                                # set the cluster
        inertia += sum(euclidean_distance(vector, centroid) ** 2 for vector in cluster)        # for each data point in cluster, calculate euclid distance, square and sum

    return inertia



def calculate_dunn_index(centroids:list[np.ndarray], clusters:list[list[np.ndarray]]) -> float:

    """
    Intra-cluster
    """
    intra_max = 0.0                                                                            # init max distance for data points inside clusters
    for cluster in clusters:                                                                   # for each cluster
        for i in range(len(cluster)):                                                          # for each vector pair in this cluster
            for j in range(i + 1, len(cluster)):
                distance = euclidean_distance(cluster[i], cluster[j])                          # calculate euclid distance
                intra_max = max(intra_max, distance)                                           # update max value if larger

    """
    Inter-cluster
    """
    inter_min = float("inf")                                                                   # init min distance between clusters/centroids
    for i in range(len(centroids)):                                                            # for each centroid pair
        for j in range(i + 1, len(centroids)):
            distance = euclidean_distance(centroids[i], centroids[j])                          # calculate euclid distance
            inter_min = min(inter_min, distance)                                               # update min value if smaller

    if intra_max == 0:                                                                         # dont divide by zero
        return 0.0

    return inter_min / intra_max