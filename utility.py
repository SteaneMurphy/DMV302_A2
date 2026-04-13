import csv
import numpy as np


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


def parse_numeric(rows:list[str]):
    """
    Parses rows of purely numeric values.

    Args:
        rows: array of CSV rows

    Returns:
        np.ndarray (float): array of floats
    """
    return np.array([[float(x.strip()) for x in row] for row in rows])


def parse_features_labels(rows:list[str]):
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


def parse_transactions(rows:list[str]):
    """
    Parses rows into transactions for association analysis.

    Args:
        rows: array of CSV rows

    Returns:
        list[list[str]]: list of transactions
    """
    return [[item.strip() for item in row] for row in rows]