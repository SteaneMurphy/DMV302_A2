import numpy as np
from sklearn.neural_network import MLPClassifier
from utility.util import read_csv, parse_features_labels, standardise

class NNClassifier:
    def __init__(self):
        self.model = MLPClassifier(                                                                
            hidden_layer_sizes=(32,),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=2000,
            random_state=42,
            shuffle=True
        )

        self.mean = None
        self.std = None

    def preprocess_data(self, path:str):
        rows = list(read_csv(path, skip_header=True))                                              # import rows from CSV
        features, labels = parse_features_labels(rows)                                             # parse rows into numeric data format, separate labels column from data
        features, mean, std = standardise(features)

        self.mean = mean                                                                           # store mean
        self.std = std                                                                             # store standard deviation

        return features, labels

    def train_model(self):
        training_data, training_labels = self.preprocess_data(                                     # preprocess training data
            "data/at_risk_students_training.csv"
        )
        self.model.fit(training_data, training_labels)                                             # fit training data to NN model

# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html