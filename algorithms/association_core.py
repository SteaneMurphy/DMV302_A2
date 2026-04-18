import numpy as np
from utility import read_csv, parse_transactions

def preprocess_data(path:str):
    rows = read_csv(path, skip_header=True)                                                    # import rows from CSV
    transactions = parse_transactions(rows)                                                    # parse rows into transaction lists (sets)

    return transactions



def calculate_support(itemset: set[str], data:list[set[str]]) -> float:
    count = 0

    for transaction in data:                                                                   # for each transaction in dataset
        if itemset.issubset(transaction):                                                      # increment counter if item set is found
            count += 1
    
    return count / len(data)                                                                   # support formula (itemset occurance / transactions)



def calculate_confidence(itemset_a: set[str], itemset_b: set[str], data:list[set[str]]) -> float:
    combined = itemset_a | itemset_b                                                           # combine item sets

    combinedSupport = calculate_support(combined, data)                                        # calculate support for item sets A u B
    singleSupport = calculate_support(itemset_a, data)                                         # calculate support for item set A
    
    if singleSupport == 0.0:                                                                   # edge-case (division by zero)
        return 0.0

    return combinedSupport / singleSupport                                                     # confidence formula (support A u B / support A)
    