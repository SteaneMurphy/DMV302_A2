import csv
import numpy as np

"""
Loads data from a CSV file into a NumPy array.

Designed for numeric datasets where each row contains
comma-separated float values.

Args:
    path (str): filepath to CSV
    skip_header (bool): whether to skip first row

Returns:
    np.ndarray: 2D array of floats
"""
def load_numeric_csv(path:str, skip_header:bool=False):
    data = []

    with open(path, 'r') as file:
        reader = csv.reader(file)

        if skip_header:
            next(reader)

        for row in reader:
            if len(row) == 0:
                continue
            data.append([float(x.strip()) for x in row])        # convert each value to float and remove whitespace

    return np.array(data)