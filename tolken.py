#Libraries
import numpy as np
import nltk
#from nltk.tokenize import TweetTokenizer

def replace_words(corpus):
    
    
    new_corpus_string=[]
    tknzr = nltk.tokenize.TweetTokenizer()
    new_corpus=[]
    
    
    for line in corpus:
        vec_of_strings=tknzr.tokenize(line)
        exclamation=False
        hashtag=False
        number=False
        cleaned_tweet=[]
        retweet=False
        for i, word in enumerate(vec_of_strings):
            if word=='rt':
                retweet=True
            elif word == "<3":
                cleaned_tweet.append('heart')
            elif i>0 and word=='_' and vec_of_strings[i-1]=='^' and vec_of_strings[i+1]=='^':
                cleaned_tweet.append('eyesmiley')
            elif word=='!':
                exclamation=True
            elif word[0]=='#':
                hashtag=True
                cleaned_tweet.append(word[1:])
            elif word[-3:]=='n\'t':
                #cleaned_tweet.append(word[:-3])
                cleaned_tweet.append('not')
            elif word.isdigit():
                number=True
            elif word!= '^' and word!=',' and word!='.':
                cleaned_tweet.append(word)
            
        if exclamation:
            cleaned_tweet.append('exclamation')
        if hashtag:
            cleaned_tweet.append('hashtag')
        if number:
            cleaned_tweet.append('thereisanumber')
        
        #if retweet==False:
   
        new_words = ' '.join(cleaned_tweet)
        new_words = new_words.encode('utf-8')
        new_corpus.append(new_words)
    return new_corpus
    