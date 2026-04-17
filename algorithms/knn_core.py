import numpy as np
from utility import read_csv, parse_features_labels, standardise, euclidean_distance

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
# def classification_knn(k_value:int):
#     training_data, training_labels = preprocess_data("data/at_risk_students_training.csv")
#     test_data, test_labels = preprocess_data("data/at_risk_students_test.csv")

#     all_data = np.vstack([training_data, test_data])
#     all_data = standardise(all_data)

#     training_data = all_data[:len(training_data)]
#     test_data = all_data[len(training_data):]

#     print("Baseline majority class:", max(set(training_labels), key=list(training_labels).count))

#     classified_data = []
#     for test_vec in test_data:                                                                    
#         distances = [
#             (euclidean_distance(test_vec, train_vec), label)
#             for train_vec, label in zip(training_data, training_labels)
#         ]      

#         distances.sort(key=lambda x: x[0])
#         neighbours = distances[:k_value]

#         weights = {}

#         for distance, label in neighbours:
#             weighting = 1 / (distance**2 + 1e-9)
#             weights[label] = weights.get(label, 0) + weighting
        
#         prediction = max(weights, key=weights.get)
#         classified_data.append(prediction)

#     return classified_data, test_labels

def classification_knn(k_value:int):
    training_data, training_labels = preprocess_data("data/at_risk_students_training.csv")
    test_data, test_labels = preprocess_data("data/at_risk_students_test.csv")

    all_data = np.vstack([training_data, test_data])
    all_data = standardise(all_data)

    training_data = all_data[:len(training_data)]
    test_data = all_data[len(training_data):]

    print("Baseline majority class:", max(set(training_labels), key=list(training_labels).count))

    classified_data = []

    for test_vec in test_data:

        # compute distances to ALL training points
        distances = [
            (euclidean_distance(test_vec, train_vec), label)
            for train_vec, label in zip(training_data, training_labels)
        ]

        # split distances + labels
        distance_values = np.array([d for d, _ in distances])
        label_values = np.array([l for _, l in distances])

        # avoid division issues / normalisation stability
        total_distance = np.sum(distance_values) + 1e-9

        # weights as per brief
        weights = distance_values / total_distance

        score = {}

        # accumulate weighted votes per class
        for weight, label in zip(weights, label_values):
            score[label] = score.get(label, 0) + weight

        prediction = max(score, key=score.get)
        classified_data.append(prediction)

    return classified_data, test_labels


def evaluate_test_set():
    # evaluate test set
    return None