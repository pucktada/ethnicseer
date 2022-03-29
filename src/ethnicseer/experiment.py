from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .ethnic_classifier import EthnicClassifier

# dir_path = 'data/names'
def load_from_directory(dir_path: str):
    data_path  = Path(dir_path)
        
    names = []
    ethnics = []

    for file_path in data_path.glob('*'):
        with open(file_path, 'r') as f:
            for line in f:
                name = line.strip()
                names += [name]
                ethnics += [file_path.name]
    return names, ethnics

def train_model(names: list, ethnics: list):
    ec = EthnicClassifier()
    train_acc = ec.fit(names, ethnics)

    ethnics_hat = ec.classify_names(names)
    print(classification_report(ethnics, ethnics_hat, 
        target_names=ec.ethnicity_classes()))

    return ec

def train_and_evaluate(names: list, ethnics: list):
    names_train, names_test, ethnics_train, ethnics_test = train_test_split(
        names, ethnics, test_size=0.3, random_state=10)

    ec = EthnicClassifier()
    train_acc = ec.fit(names_train, ethnics_train)

    ethnics_hat = ec.classify_names(names_test)

    correct_count = np.sum([(a==b) for a, b in zip(ethnics_test, ethnics_hat)])
    total_count = len(ethnics_test)
    test_acc = float(correct_count)/total_count

    print('train acc:', train_acc)
    print('test acc:', test_acc)
    print(classification_report(ethnics_test, ethnics_hat, 
        target_names=ec.ethnicity_classes()))
