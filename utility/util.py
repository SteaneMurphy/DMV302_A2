import csv
import numpy as np

###########################
#       CSV UTILITY       #
###########################

def read_csv(path:str, skip_header:bool = False):
    """
    Reads a CSV file and returns non-empty rows.

    Args:
        path (str): filepath to CSV
        skip_header (bool): whether to skip first row

    Returns:
        list[str]: a single CSV row as a list of strings
    """
    with open(path, 'r') as file:                                                   # open csv file
        reader = csv.reader(file)                                                   # read file

        if skip_header:                                                             # optional skip first row
            next(reader)

        for row in reader:                                                          # for each row return row if not empty
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
    return np.array([[float(x.strip()) for x in row] for row in rows])              # strip and convert to float for each row in given dataset


def parse_features_labels(rows:list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses rows into features and labels.

    Assumes last column is an integer class label.

    Args:
        rows: array of CSV rows

    Returns:
        np.ndarray (float): features
        np.ndarray (int): labels 
    """
    features = [[float(x.strip()) for x in row[:-1]] for row in rows]               # strip and convert to float for each row in given dataset (minus label column)
    labels = [int(row[-1].strip()) for row in rows]                                 # strip and convert to int for each label column per row
    return np.array(features), np.array(labels)



def parse_transactions(rows:list[str]) -> list[set[str]]:
    """
    Parses rows into transactions for association analysis.

    Args:
        rows: array of CSV rows

    Returns:
        list[set[str]]: list of transactions
    """
    return [{item.strip().lower() for item in row} for row in rows]                 # strip and return each row as a set for each row in dataset


###########################
#  PREPROCESSING UTILITY  #
###########################

def calculate_mean(features: np.ndarray) -> np.ndarray:
    return np.mean(features, axis=0)

def calculate_standard_deviation(features: np.ndarray) -> np.ndarray:
    return np.std(features, axis=0)

def standardise(features: np.ndarray) -> np.ndarray:
    """
    Standardises the feature data so each column has mean 0 and standard deviation 1.

    Args:
        features (np.ndarray): 2D array where each row is a sample and each column is a feature

    Returns:
        np.ndarray: standardised features
    """
    mean = calculate_mean(features)                                                 # calculate mean
    std = calculate_standard_deviation(features)                                    # calculate standard deviation
    std = np.where(std == 0, 1, std)                                                # edge-case, division by 0

    features = (features - mean) / std                                              # standardise data (z-score scaling)    
    return features, mean, std                                          

###########################
#      MATH UTILITY       #
###########################

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
    return np.linalg.norm(point_a - point_b)                                        # numpy euclidean distance formula


###########################
#   ASSOCIATION UTILITY   #
###########################
                           
def parse_sup_command(input: str) -> set[str]:
    """
    Parses a support command into an itemset.

    The input string is expected to start with the keyword 'sup' followed
    by one or more comma-separated item names.

    Args:
        input (str): Raw user input beginning with 'sup'

    Returns:
        set[str]: A set of cleaned, lowercase item names
    """
    input = input.replace("sup", "", 1).strip()

    return {
        item.strip().lower()
        for item in input.split(",")
        if item.strip()
    }



def parse_con_command(input: str) -> tuple[set[str], set[str]]:
    """
    Parses a confidence command into two itemsets representing
    the left-hand side and right-hand side of the input.

    The input string is expected to start with the keyword 'con' and
    use '-->' to separate itemsets.

    Args:
        input (str): Raw user input beginning with 'con'.

    Returns:
        tuple[set[str], set[str]]:
            - First set (left side)
            - Second set (right side)
    """
    input = input.replace("con", "", 1).strip()

    left, right = input.split("-->")

    itemA = {item.strip().lower() for item in left.split(",") if item.strip()}
    itemB = {item.strip().lower() for item in right.split(",") if item.strip()}

    return itemA, itemB



###########################
#   SHOPPING MART TEXT    #
###########################

header = """
┌────────────────────────────────────────────┐
│              SHOPPING MART                 │
│        Market Basket Analysis Tool         │
└────────────────────────────────────────────┘"""



commands = """----------------------------------------------
COMMANDS

1. sup item[,item]                         # calcuates support #

2. con item[,item] --> item[,item]         # calculates confidence #

3. exit                                    # quits the application #
"""



def display_result(itemset_a: set[str], result: float, association_type: str = "support", itemset_b: set[str] = None):
    if itemset_b is None:
        items = ", ".join(sorted(itemset_a))
        print(f"\n{items} has a {association_type} of {result:.4f} ({result * 100:.2f}%)".upper())
    else:
        left = ", ".join(sorted(itemset_a))
        right = ", ".join(sorted(itemset_b))
        print(f"\n{left} --> {right} has a {association_type} of {result:.4f} ({result * 100:.2f}%)".upper())