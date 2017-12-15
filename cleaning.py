import os
from gensim import corpora
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pickle 
from sklearn.feature_extraction.text import TfidfVectorizer
def get_dynamic_stopwords(corpus, MinDf, MaxDf,sublinearTF=True,useIDF=False):
    vectorizer = TfidfVectorizer(
            min_df = MinDf,   # if between 0 and 1: percentage, if int: absolute value.
            max_df = MaxDf,   # # if between 0 and 1: percentage, if int: absolute value. ( more then 0.8 * number of tweets )
            sublinear_tf = sublinearTF, # scale the term frequency in logarithmic scale
            use_idf = useIDF
            )
    vectorizer.fit_transform(corpus)
    return vectorizer.stop_words_
    


def remove_stopwords(corpus, custom_stop_words):
    """
    denne ble stygg as fuck, men vet ikke helt om jeg får den penere... 
    """
    
    new_corpus=[]
    for line in corpus:
        words=line.decode("utf-8")#.split(" ")
        words = ''.join(words.rsplit('\n', 1))
        words=words.lower().split(" ")
        new_words=[]
        for word in words: 
            if word not in custom_stop_words:
                new_words.append(word)
               
            
        new_words = ' '.join(new_words)
        new_words = new_words.encode('utf-8')
        new_corpus.append(new_words) 
    return new_corpus



### De under her bruker vi ikke mer...

def create_dictionary(cluster_file):
    """ 
    Input: 
    cluster_file: name of the file that contains the information about the clusters. The function assums that the file 
    is structures exactly as the file 50mpaths2.txt which can be dowloaded from 
    "http://www.cs.cmu.edu/~ark/TweetNLP/#parser_down". 
    
    Output: a dictionary, with words as key and cluster-id as value. 
    """
    cluster_dictionary = {}
    with open(cluster_file,'rb') as infile:
        for line in infile:
            # decode from Bytes too string
            string = line.decode("utf-8")
            # because there are words with '\n' in it we must just remove the last occurence ex: 101011\tarm\n\t42
            string = ''.join(string.rsplit('\n', 1))
            # because of cases where the word contains '\t' we must just remove the first and last '\t' ex: "11010101\t\t\t42"
            string = string.split('\t',1)
            split_last_part = string[1].rsplit('\t', 1)
            string[1] = split_last_part[0]
            string.append(split_last_part[1])
            #setting the word as key in dictionary and the cluster-id as value.
            cluster_dictionary.update({string[1]:string[0]})
            #pickle.dump(cluster_dictionary, open( "cluster_dictionary.pkl", "wb" ) )
    return cluster_dictionary 

def create_clusterized_corpus(corpus, cluster_dictionary):
    """ 
    Input: 
        corpus: a corpus as created by create_coprus 
        cluster_dictionary: A dictionary created using the function creat_dictionary. 
    Output: 
        new_corpus: A new corpus on the same format as the original one, where all words in the dictionary are replaced
        by the cluster id. 
    """
    new_corpus=[]
    for line in corpus:
        words=line.decode("utf-8")#.split(" ")
        words = ''.join(words.rsplit('\n', 1))
        words=words.split(" ")
        for i,word in enumerate(words) : 
            value=cluster_dictionary.get(word)
            if value is not None:
                words[i]=value
        words = ' '.join(words)
        words = words.encode('utf-8')
        new_corpus.append(words) 
    return new_corpus
    
