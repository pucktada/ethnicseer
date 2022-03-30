# ethnicseer ['ethnic-seer'] - a name-ethnicity classifier

*ethnicseer* ('ethnic-seer') is a name-ethnicity classifier, written in python. It can determine the ethnicity of a given name, using linguistic features such as sequences of characters found in the name and its phonetic pronounciation. *ethnicseer* comes with a pre-trained model, which can handle the following 12 ethnicities: middle-eastern, chinese, english, french, vietnam, spanish, italian, german, japanese, russian, indian, and korean. The included pre-trained model can achieve around 84% accuracy on the test data set.

*ethnicseer* is based on the name-ethnicity classifier, orginally proposed here:
> Treeratpituk, Pucktada, and C. Lee Giles. "Name-ethnicity classification and ethnicity-sensitive name matching." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 26. No. 1. 2012.

> Paper URL : https://ojs.aaai.org/index.php/AAAI/article/download/8324/8183

# Requirements
* abydos
* scikit-learn
* nltk
* python = 3.9+

# Installation

`ethnicseer` can be installed using `pip` 

```
pip install ethnicseer
```

# Usages

Once installed, you can use `ethnicseer` within your python code to classify whether a Thai name is a person name or a corporate name. 

```
>>> from ethnicseer import EthnicClassifier

>>> ec = EthnicClassifier.load_pretrained_model()
>>> ec.classify_names(['Yūta Nakayama','Marcel Halstenberg','Raphaël Varane'])
['jap', 'ger', 'frn']
```

```
>>> ec = EthnicClassifier.load_pretrained_model()
>>> ec.classify_names(['Yūta Nakayama','Marcel Halstenberg','Raphaël Varane'])
['jap', 'ger', 'frn']
```

## Citation

```
Treeratpituk, Pucktada, and C. Lee Giles. "Name-ethnicity classification and ethnicity-sensitive name matching." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 26. No. 1. 2012.
```

## Author
Pucktada Treeratpituk, Bank of Thailand (pucktadt@bot.or.th)

## License

This project is licensed under the Apache Software License 2.0 - see the [LICENSE](LICENSE) file for details

