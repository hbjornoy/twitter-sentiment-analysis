#Libraries
import numpy as np
import nltk
#from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer

from nltk.tokenize import TweetTokenizer
def replace_words(corpus):
    
    new_corpus=[]
    new_corpus_string=[]
    
    tknzr = TweetTokenizer(reduce_len=True)
    ps = PorterStemmer()
    
    
    
    for line in corpus:
        words=tknzr.tokenize(line)
        exclamation=False
        hashtag=False
        number=False
        cleaned_tweet=[]
        retweet=False
        for i, word in enumerate(words):
            #If we want to know if tweets are a retweet:
            if word=='rt':
                retweet=True
            
           #Want to look for special signs
            elif word=='!':
                exclamation=True
                cleaned_tweet.append(word)
            elif word[0]=='#':
                hashtag=True
                cleaned_tweet.append(word[1:])
                
            
            #want to look for abbreviations that mean not:
            elif word[-3:]=='n\'t':
                #cleaned_tweet.append(word[:-3])
                cleaned_tweet.append('not')
            
            #Want to look for numbers
            elif word.isdigit():
                number=True
            
            #want to find hearts, hugs, kisses, etc: 
            elif word == "<3":
                cleaned_tweet.append('heart')
            elif (word == "xoxo" or word == "xo" or word == "xoxoxo" or word == "xxoo"):
                cleaned_tweet.append('hug')  
                cleaned_tweet.append('kiss')
            elif (word=='xx' or word=='xxx'):
                cleaned_tweet.append('kiss')
                
            
            #looking for different types of smilies that have not been removed from the dataset: 
            elif i>0 and word=='_' and words[i-1]=='^' and words[i+1]=='^':
                cleaned_tweet.append('eyesmiley')
            elif i>0 and word=='o' and words[i-1]==':':
                cleaned_tweet.append('openmouthface')
            elif i>0 and word=='/' and words[i-1]==':':
                cleaned_tweet.append('slashsmiely')
            elif i>0 and word=='*' and (words[i-1]==':' or words[i-1]==';'):
                cleaned_tweet.append('kiss')
            elif i>0 and word==')' and (words[i-1]==':' or words[i-1]==';'):
                cleaned_tweet.append('possmiley')
            elif i>0 and word=='(' and (words[i-1]==':' or words[i-1]==';'):
                cleaned_tweet.append('negsmiley')
            elif i>0 and words[i-1]=='(' and (word==':' or word==';'):
                cleaned_tweet.append('possmiley')
            elif i>0 and words[i-1]==')' and (word==':' or word==';'):
                cleaned_tweet.append('negsmiley')
            elif i>0 and (word=='d' and (words[i-1]==':' or words[i-1]==';')) or word == ':d':
                cleaned_tweet.append('possmiley')
            elif i>0 and (word=='p' and (words[i-1]==':' or words[i-1]==';')) or word == ':p':
                cleaned_tweet.append('possmiley')
            elif i>0 and word=='d' and (words[i-1]==':' or words[i-1]==';' or words[i-1]=='x'):
                cleaned_tweet.append('possmiley')
        
        
            elif (word!= '^' and word!=',' and word!='.' and word!=':' and word!='-' and word!='´' and word!=';'
                 and word!=')' and word!='(' and word!='*'):
            
                cleaned_tweet.append(ps.stem(word))
            
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


