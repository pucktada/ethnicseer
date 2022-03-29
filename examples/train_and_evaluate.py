from nameseer import NameClassifier
from nameseer import load_from_directory, train_model, train_and_evaluate

names, cats = load_from_directory('data/small_companies')
print(names[:10])
print(cats[:10])
train_and_evaluate(names, cats)