import csv
import numpy as np


"""
Loads data from a CSV file into a numpy array.

Designed for numeric datasets where each row contains
comma-separated float values.

Args:
    path (str): filepath to CSV
    skip_header (bool): whether to skip first row

Returns:
    np.ndarray: 2D array of floats
"""
def load_numeric(path:str, skip_header:bool=False):                           # Q1
    data = []

    with open(path, 'r') as file:
        reader = csv.reader(file)

        if skip_header:
            next(reader)

        for row in reader:
            if len(row) == 0:
                continue
            # convert each value to float and remove whitespace
            data.append([float(x.strip()) for x in row])        

    return np.array(data)


"""
Loads data from a CSV file into two numpy arrays.

Designed for supervised numeric datasets where each row contains
comma-separated feature values followed by a label.

Args:
    path (str): filepath to CSV
    skip_header (bool): whether to skip first row

Returns:
    np.ndarray: feature matrix (2D float array)
    np.ndarray: label vector (1D int array)
"""
def load_features_labels(path:str, skip_header:bool=False):                    # Q3
    features, labels = [], []

    with open(path, 'r') as file:
        reader = csv.reader(file)

        if skip_header:
            next(reader)

        for row in reader:
            if len(row) == 0:
                continue
            # extract all columns except last, convert to float and remove whitespace
            features.append([float(x.strip()) for x in row[:-1]])
            # extract last column as target label, convert to int and remove whitespace  
            labels.append(int(row[-1].strip()))

    return np.array(features), np.array(labels)


"""
Loads data from a CSV file and splits into a list of transactions.

Args:
    path (str): filepath to CSV
    skip_header (bool): whether to skip first row

Returns:
    list[list (str)]: inner list contains a transaction row
"""
def load_transactions(path:str, skip_header:bool=False):                       # Q4
    transactions = []

    with open(path, 'r') as file:
        reader = csv.reader(file)

        for row in reader:
            if len(row) == 0:
                continue

            # add row to list, strip whitespace
            transactions.append([item.strip() for item in row])

    return transactions
        