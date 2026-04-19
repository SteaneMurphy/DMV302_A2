import numpy as np
from utility.util import read_csv, parse_features_labels

def preprocess_data(path:str, mean, std):
    rows = list(read_csv(path, skip_header=True))                                              # import rows from CSV
    features, labels = parse_features_labels(rows)                                             # parse rows into numeric data format, separate labels column from data
    features = (features - mean) / std                                                         # standardise test data using training data mean/std

    return features, labels

def validate_model(nn_instance):
    test_data, test_labels = preprocess_data(
        "data/at_risk_students_test.csv",
        nn_instance.mean, 
        nn_instance.std
    )

    label_predict = nn_instance.model.predict(test_data)                                       # prediction algorithm (sklearn) against test data set
    accuracy = np.mean(label_predict == test_labels)                                           # calculate accuracy of predictions against acutal labels
    error = 1 - accuracy                                                                       # calculate error rate based on accuracy

    return accuracy, error