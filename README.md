### Project Text Sentiment Classification DESCRIPTION

The task of this project is to predict if a tweet message used to contain a positive :) or negative :( smiley, by considering only the remaining text! 

### How to produce the prediction ( run.py )

Fork or download zip of repository, and run the `run.py` from the root folder with `Python run.py`. This pruduces the file `powerpuffz_kagglescore.csv` in the root folder. 

### Installations and requirements

In order to run `run.py` some packages with their dependencies are required. We recommend creating a clean Anaconda Environment, and installing the following packages via Anaconda Navigator:

- Python V: 3.6.3
- Numpy V: 1.12.1
- Keras V: 2.0.8 
- Tensorflow V: 1.2.1
- Scikit-learn V: 0.19.1
- Gensim  V: 3.1.0

In order to run the whole preprocessing and other project components, the following packages with their dependencies are required:

- NLTK V: 3.2.5 
- Ekphrasis V: 0.3.6 
  - Run: pip install ekphrasis. 
  - NB: Will take some time on first import to download "Word statistics files". 
- Enchant V: 2.0.0 ( mac / linux only ) 
  - Run: pip install pyenchant
  - NB: No version for Python on 64-bit windows. Run on a OSX/Linux-machine.
  - NB: This is only used for the enchant word-dictionary, and can be exchanged with another dictionary if running on windows is wanted.
  
### Files downloaded and used in this project

All downloaded files should be placed in the root project folder.

- Download Glove word vectors word vecs from [Standford Twitter Glove]("http://nlp.stanford.edu/data/glove.twitter.27B.zip")
- Download and unzip in root folder the training and test dataset from [Kaggle]("https://www.kaggle.com/c/epfml17-text/data")

