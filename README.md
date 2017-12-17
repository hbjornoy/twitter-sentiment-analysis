## Installations 

# Gensim for windows
Download gensim through pip ( python package manager ), make sure to run with administrator privleges. 
`pip install --upgrade gensim`
`conda install -c conda-forge keras tensorflow` 

pip install ekphrasis
pip install pyenchant

# GENSIM  
- V: 

# KERAS 
- V:

# TENSORFLOW 
- V: 1.2.1

# SKLEARN 

# NLTK

# Enchant 
- Only used for dictionary, NB: difficult to install on windows. 

# Ekphrasis


## Downloading files 

# Downloading gensim word vecs?  

## Explain RUN file 



#PLAN: 
- Make new anaconda environment and find needed packages 
- Write about it



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
