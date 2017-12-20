### Installations and requirements

In order to run `RUN.py` some packages with their dependencies are required. We recommend creating a clean Anaconda Environment, and installing the following packages via Anaconda Navigator:

- Python V: 3.6.3
- Numpy V: 1.12.1
- Keras V: 2.0.8 
- Tensorflow V: 1.2.1
- Scikit-learn V: 0.19.1
- Gensim  V: 3.1.0

In order to run the whole preprocessing and other project components, the following packages with their dependencies are required:

- NLTK V: 3.2.5 

- Enchant V: 2.0.0 ( mac / linux only ) 
  - Run: pip install pyenchant
  - NB: No version for Python on 64-bit windows. Run on a OSX/Linux-machine.
  - NB: This is only used for the enchant word-dictionary, and can be exchanged with another dictionary if running on windows is wanted.
 
- Ekphrasis V: 0.3.6 
  - Run: pip install ekphrasis. 
  - NB: Will take some time on first import to download "Word statistics files". 


### Downloadable files needed.

All downloaded files should be placed in the root project folder.

- Download Glove word vectors word vecs from [Standford Twitter Glove]("http://nlp.stanford.edu/data/glove.twitter.27B.zip")
- Download training and test dataset from [Kaggle]("https://www.kaggle.com/c/epfml17-text/data")


### Explain RUN file 

Fork or download zip of repository, and run the `run.py` from terminal with `Python run.py`. This pruduces the file `powerpuffz_kagglescore.csv` in the root folder. 



# Project Text Sentiment Classification DESCRIPTION


The task of this competition is to predict if a tweet message used to contain a positive :) or negative :( smiley, by considering only the remaining text.

As a baseline, we here provide sample code using word embeddings to build a text classifier system.

Submission system environment setup:

1. The dataset is available from the Kaggle page, as linked in the PDF project description

 Download the provided datasets `twitter-datasets.zip`.

2. To submit your solution to the online evaluation system, we require you to prepare a “.csv” file of the same structure as sampleSubmission.csv (the order of the predictions does not matter, but make sure the tweet ids and predictions match). Your submission is evaluated according to the classification error (number of misclassified tweets) of your predictions.

*Working with Twitter data:* We provide a large set of training tweets, one tweet per line. All tweets in the file train pos.txt (and the train pos full.txt counterpart) used to have positive smileys, those of train neg.txt used to have a negative smiley. Additionally, the file test data.txt contains 10’000 tweets without any labels, each line numbered by the tweet-id.

Your task is to predict the labels of these tweets, and upload the predictions to kaggle. Your submission file for the 10’000 tweets must be of the form `<tweet-id>`, `<prediction>`, see `sampleSubmission.csv`.

Note that all tweets have already been pre-processed so that all words (tokens) are separated by a single whitespace. Also, the smileys (labels) have been removed.
