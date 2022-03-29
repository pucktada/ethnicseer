from ethnicseer import EthnicClassifier

ec = EthnicClassifier.load_pretrained_model()
print('possible classes: ')
print(ec.ethnicity_classes())
print('clssifing names: Yūta Nakayama, Marcel Halstenberg, Raphaël Varane')
print(ec.classify_names(['Yūta Nakayama','Marcel Halstenberg','Raphaël Varane']))
