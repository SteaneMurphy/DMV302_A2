import numpy as np
from sklearn.neural_network import MLPClassifier
from utility.util import read_csv, parse_features_labels, standardise, calculate_mean, calculate_standard_deviation

class NNClassifier:
    def __init__(self):
        self.model = MLPClassifier(                                                                
            hidden_layer_sizes=(32, 16),
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

        self.mean = mean
        self.std = std

        return features, labels

    def preprocess_data_test(self, path:str):
        rows = list(read_csv(path, skip_header=True))                                              # import rows from CSV
        features, labels = parse_features_labels(rows)                                             # parse rows into numeric data format, separate labels column from data
        features = (features - self.mean) / self.std

        return features, labels

    def train_model(self):
        # training_data, training_labels = self.preprocess_data("data/at_risk_students_training.csv")
        training_data, training_labels = self.preprocess_data("data/corrected_training.csv")
        # print(f"NPMEAN Y-TRAIN: {np.mean(training_labels)}")
        # print(f"SCALING SANITY CHECK: {np.std(training_data, axis=0)}")
        self.model.fit(training_data, training_labels)

    def test_model(self):
        # test_data, test_labels = self.preprocess_data_test("data/at_risk_students_test.csv")
        test_data, test_labels = self.preprocess_data_test("data/corrected_test.csv")

        label_predict = self.model.predict(test_data)
        accuracy = np.mean(label_predict == test_labels)
        error = 1 - accuracy

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Error: {error:.4f}")
