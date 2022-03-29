from nameseer import NameClassifier

nc = NameClassifier.load_pretrained_model()
print('possible classes: ')
print(nc.name_classes())
print('clssifing names: ประยุทธ์ จันทร์โอชา, แอดวานซ์อินโฟร์เซอร์วิส')
print(nc.classify_names(['ประยุทธ์ จันทร์โอชา','แอดวานซ์อินโฟร์เซอร์วิส']))
