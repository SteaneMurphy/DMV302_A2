import random
from numpy import mean
from utility import read_csv, parse_numeric, standardise, euclidean_distance

def preprocess_data(path:str):
    rows = read_csv(path, skip_header=True)                                                    # import rows from CSV
    features = parse_numeric(rows)                                                             # parse rows into numeric data format
    features = standardise(features)                                                           # standardise the rows for clustering

    return features

def clustering_k_means(k_value:int, tolerance:float, max_iterations:int):
    data = preprocess_data("/data/household_wealth.csv")                                       # preprocess data - standardise

    centroids = random.sample(data, k_value)                                                   # choose centroids, k-means

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

        if max(shifts) < tolerance:                                                            # check largest centroid movement is below tolerance
            break                                                                              # end algorithm if true

        centroids = new_centroids                                                              # store reference to new centroids                                                   

    return centroids, clusters

    #https://numpy.org/doc/stable/reference/generated/numpy.mean.html