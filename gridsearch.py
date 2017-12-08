# Imports and stuff
import os
import time
import csv

#Libraries
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction import stop_words
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold


def gridsearch(corpus,labels,nr_lines_total):
    min_df_list = list(range(5,13))
    max_df_list = np.linspace(0.5, 1.6, num=10)

    scores = []
    best_score = 0
    best_min_df = 0 
    best_max_df = 0

    for min_df in min_df_list:
        for max_df in max_df_list:

            #Creating K-fold for cross validation
            kf = StratifiedKFold(n_splits=2)

            totalsvm = 0

            for train_index, test_index in kf.split(corpus,labels):

                X_train = [corpus[i] for i in train_index]
                X_test = [corpus[i] for i in test_index]

                y_train, y_test = labels[train_index], labels[test_index]

                vectorizer = TfidfVectorizer(
                    min_df = min_df, # removing word that occure less then X times 
                    max_df = max_df, # remove words that are too frequent ( more then 0.8 * number of files )
                    sublinear_tf=True, # scale the term frequency in logarithmic scale
                    use_idf =False
                    #stop_words = 'english', # Removing stop-words
                )

                train_corpus_tf_idf = vectorizer.fit_transform(X_train) 
                test_corpus_tf_idf = vectorizer.transform(X_test)

                model1 = LinearSVC()
                model1.fit(train_corpus_tf_idf,y_train)
                result1 = model1.predict(test_corpus_tf_idf)


                #totalMatSvm = totalMatSvm + confusion_matrix(y_test, result1)
                totalsvm = totalsvm+sum(y_test==result1)

            score = totalsvm / nr_lines_total
            scores.append(score)

            if(score > best_score):
                best_min_df = min_df
                best_max_df = max_df
                best_score = score

                print("Best score: ", best_score, "\n",
                      "With min_df:", min_df, " and ",
                      "Max_df:", max_df)