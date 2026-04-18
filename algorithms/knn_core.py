import numpy as np
from utility.util import read_csv, parse_features_labels, standardise, euclidean_distance

def preprocess_data(path:str):
    rows = list(read_csv(path, skip_header=True))                                              # import rows from CSV
    features, labels = parse_features_labels(rows)                                             # parse rows into numeric data format, separate labels column from data
    # features = standardise(features)                                                           # standardise the rows for clustering

    return features, labels

"""
1NN: K = 1. Check nearest neighbour
nNN/KNN: K = variable. Check 'K' nearest neighbours
Weighted KNN: closest neighbours have higher weight on result
"""
import numpy as np
from utility.util import read_csv, parse_features_labels, standardise, euclidean_distance

def preprocess_data(path: str):
    rows = list(read_csv(path, skip_header=True))
    features, labels = parse_features_labels(rows)
    return features, labels


def classification_knn(k_value: int):

    training_data, training_labels = preprocess_data("data/at_risk_students_training.csv")
    test_data, test_labels = preprocess_data("data/at_risk_students_test.csv")

    mean = np.mean(training_data, axis=0)
    std = np.std(training_data, axis=0)
    std = np.where(std == 0, 1, std)

    training_data = (training_data - mean) / std
    test_data = (test_data - mean) / std

    classified_data = []

    for test_vec in test_data:

        distances = [
            (euclidean_distance(test_vec, train_vec), label)
            for train_vec, label in zip(training_data, training_labels)
        ]

        distances.sort(key=lambda x: x[0])
        neighbours = distances[:k_value]

        weights = {}

        for distance, label in neighbours:
            weight = 1 / (distance**2 + 1e-9)
            weights[label] = weights.get(label, 0) + weight

        prediction = max(weights, key=weights.get)
        classified_data.append(prediction)

    return classified_data, test_labels

def evaluate_test_set():
    # evaluate test set
    return None