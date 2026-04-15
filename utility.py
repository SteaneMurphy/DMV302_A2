import csv
import numpy as np
from typing import List, Tuple

def read_csv(path:str, skip_header:bool = False):
    """
    Reads a CSV file and returns non-empty rows.

    Args:
        path (str): filepath to CSV
        skip_header (bool): whether to skip first row

    Returns:
        list[str]: a single CSV row as a list of strings
    """
    with open(path, 'r') as file:
        reader = csv.reader(file)

        if skip_header:
            next(reader)

        for row in reader:
            if row:
                yield row


def parse_numeric(rows:list[str]) -> np.ndarray[float]:
    """
    Parses rows of purely numeric values.

    Args:
        rows: array of CSV rows

    Returns:
        np.ndarray (float): array of floats
    """
    return np.array([[float(x.strip()) for x in row] for row in rows])


def parse_features_labels(rows:list[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses rows into features and labels.

    Assumes last column is an integer class label.

    Args:
        rows: array of CSV rows

    Returns:
        np.ndarray (float): features
        np.ndarray (int): labels 
    """
    features = [[float(x.strip()) for x in row[:-1]] for row in rows]
    labels = [int(row[-1].strip()) for row in rows]
    return np.array(features), np.array(labels)


def parse_transactions(rows:list[str]) -> List[List[str]]:
    """
    Parses rows into transactions for association analysis.

    Args:
        rows: array of CSV rows

    Returns:
        list[list[str]]: list of transactions
    """
    return [[item.strip() for item in row] for row in rows]


# z-score scaling
def standardise(features: np.ndarray) -> np.ndarray:
    """
    Standardises the feature data so each column has mean 0 and standard deviation 1.

    Args:
        features (np.ndarray): 2D array where each row is a sample and each column is a feature

    Returns:
        np.ndarray: standardised features
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std = np.where(std == 0, 1, std)        # edge-case, division by 0

    return (features - mean) / std
    
    #https://numpy.org/doc/stable/reference/generated/numpy.mean.html
    #https://numpy.org/doc/stable/reference/generated/numpy.std.html
    #https://numpy.org/doc/stable/reference/generated/numpy.where.html
    #https://docs.python.org/3/library/csv.html

def euclidean_distance(point_a:np.ndarray, point_b:np.ndarray) -> float:
    """
    Calculates the Euclidean distance between two data points.

    This is used to measure similarity between feature vectors,
    where smaller values indicate closer points in feature space.

    Numpy function wrapped to provide readibility.

    Args:
        point_a (np.ndarray): first data point
        point_b (np.ndarray): second data point

    Returns:
        float: straight-line distance between the two points
    """
    return np.linalg.norm(point_a - point_b)

    #https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html