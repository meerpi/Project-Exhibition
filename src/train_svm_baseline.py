# baseline SVM classifier to see if landmarks are even separable
# spoiler: it works but accuracy is meh (~65%)

import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(csv_path):
    X, y = [], []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            label = row[0]
            features = [float(x) for x in row[2:]]
            X.append(features)
            y.append(ord(label) - ord("A"))
    return np.array(X), np.array(y)

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/landmarks_default.csv"
    X, y = load_data(csv_path)
    print(f"loaded {len(X)} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = SVC(kernel="rbf", C=10, gamma="scale")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"SVM accuracy: {acc*100:.1f}%")
    # its okay but not great, need something better
