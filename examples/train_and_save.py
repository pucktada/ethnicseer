from ethnicseer import EthnicClassifier
from ethnicseer import load_from_directory, train_model, train_and_evaluate

names, ethnics = load_from_directory('data/names')
print(names[:10])
print(ethnics[:10])
nc = train_model(names, ethnics)
nc.save_model('src/ethnicseer/ethnicseer')