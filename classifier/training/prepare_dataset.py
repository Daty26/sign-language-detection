import csv

def load_dataset(path="dataset.csv"):
    X, y = [], []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            y.append(row[0])
            X.append([float(v) for v in row[1:]])
    return X, y
