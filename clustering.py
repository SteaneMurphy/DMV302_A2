from utility import read_csv, parse_numeric, standardise

def preprocess_data(path:str):
    rows = read_csv(path, skip_header=True)                  # import rows from CSV
    features = parse_numeric(rows)                           # parse rows into numeric data format
    features = standardise(features)                         # standardise the rows for clustering

    return features

data = preprocess_data("/data/household_wealth.csv")