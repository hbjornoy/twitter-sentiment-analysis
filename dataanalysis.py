
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer


def indicate_number(corpus):
    """
    Input: A corpus
    Output: A corpus, where every line that contains a number ends with "Thereisanumber". 
    """
    new_corpus=[]
    for line in corpus:
        number_in_tweet=False
        words=line.decode("utf-8")#.split(" ")
        words = ''.join(words.rsplit('\n', 1))
        words=words.lower().split(" ")
        new_words=[]
        for word in words: 
            if word.isdigit():
                number_in_tweet=True
        if number_in_tweet:
            words.append('Thereisanumber')
       
        new_words = ' '.join(words)
        new_words = new_words.encode('utf-8')
        new_corpus.append(new_words) 

    
    
    return new_corpus

def corpus_to_strings(original_corpus):
    """ 
    Input: A corpus on 'normal' format: a list of binary strings
    Output: A corpus on string format: a list of strings
    """
    corpus=[]
    for line in original_corpus:
        words=line.decode("utf-8")#.split(" ")
        words = ''.join(words.rsplit('\n', 1))
        corpus.append(words)
    return corpus

def compare_set_char(neg_counter,pos_counter,characters):
    """
    Input: 
    neg_counter: A counter for the negative tweets: contains information about how many times each chatacter appears in the
    negative data. (from collections import Counter)
    pos_counter:  A counter for the positive tweets: contains information about how many times each chatacter appears in the
    positive data. (from collections import Counter)
    characters: A list of charactes one wants to check.
    
    Ouput: For each character; Prints information about how many percentage of the character was found in 
    the positive test data, and how large percentage of the positive tweets that contained that character. 
    """
    for char in characters:
        in_pos=int((pos_counter).get(char))
        in_neg=int((neg_counter).get(char))
        total=in_pos+in_neg
        print('percentage of ', char, 'in pos', in_pos/total, 'this is percentage in pos',in_pos/1250000)
        
def compare_sets(words,vec_pos,vec_neg):
    """
    Input:
    words: A list of words one wants to check.
    vec_pos: A counter for the positive tweets: contains information about how many times each word appears in the
    positive data(sklearn.feature_extraction.text import CountVectorizer)
    vec_neg: A counter for the negative tweets: contains information about how many times each word appears in the
    negative data (sklearn.feature_extraction.text import CountVectorizer)
    
    Output: For each word; Prints information about how many percentage of the word was found in 
    the positive test data.
    """
    for word in words:
        in_pos=int(vec_pos.vocabulary_.get(word))
        in_neg=int(vec_neg.vocabulary_.get(word))
        total=in_pos+in_neg
        print('percentage of ', word, 'in pos', in_pos/total)
        
def string_to_char(corpus):
    """
    Input:
    corpus: A corpus, consisting of strings (not binary)!
    
    Output: 
    characters: A long list consisting of all characters found in corpus 
    """
     
    characters=[]
    for line in corpus:
        for word in line:
            for char in word:
                characters.append(char)
    return characters

     