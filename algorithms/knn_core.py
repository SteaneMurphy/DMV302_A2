import numpy as np
from utility.util import read_csv, parse_features_labels, standardise, euclidean_distance

"""
1NN: K = 1. Check nearest neighbour
nNN/KNN: K = variable. Check 'K' nearest neighbours
Weighted KNN: closest neighbours have higher weight on result
"""
def preprocess_data(path: str):                                                                         
    rows = list(read_csv(path, skip_header=True))                                                       # import rows from CSV
    features, labels = parse_features_labels(rows)                                                      # parse rows into features, labels
    return features, labels



def classification_knn():

    training_data, training_labels = preprocess_data("data/at_risk_students_training.csv")              # preprocess training data
    test_data, test_labels = preprocess_data("data/at_risk_students_test.csv")                          # preprocess test data

    training_data, mean, std = standardise(training_data)                                               # standardise training data; mean/standard deviation                                                         
    std = np.where(std == 0, 1, std)                                                                    # prevents division by zero in event std is zero
    test_data = (test_data - mean) / std                                                                # standardise test data scaling by training mean/std

    classified_data = []                                                                                # container for evaluated points

    for test_vec in test_data:                                                                          # for each data point in test data
        distances = []

        for i in range(len(training_data)):                                                             # for each data point in training data
            train_vec = training_data[i]                                                                
            label = training_labels[i]
            distance = euclidean_distance(test_vec, train_vec)                                          # calculate Euclid distance between test and train point
            distances.append((distance, label))                                                         # final distances

        weights = {}                                                                                    # container for calculated weights
        total_distance = sum(d for d, l in distances)                                                   # sum the all the distances for this point

        for distance, label in distances:                                                               # for each distance calculated
            weight = distance / total_distance if total_distance != 0 else (1 / len(distances))         # apply the weighting
            weights[label] = weights.get(label, 0) + weight                                             # sum the weight per label

        prediction = max(weights, key=weights.get)                                                      # which label has the higher weight
        classified_data.append(prediction)                                                              # assign this predicted weight to this point

    return classified_data, test_labels                                                                 # return total set of predictions



def test_evaluation(classified_data: list[int], test_labels: list[int]) -> float:
    correct = 0

    for i in range(len(classified_data)):                                                               # for each test point
        if classified_data[i] == test_labels[i]:                                                        # does this data points label match the test label for the same point
            correct += 1

    return correct / len(test_labels)                                                                   # calcuate accuracy