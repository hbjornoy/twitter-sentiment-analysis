import numpy as np
import nltk
#from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer

from nltk.tokenize import TweetTokenizer



def preprocess_corpus(corpus,stemming=False,
                      all_smilies=False, pos_smilies=False, neg_smilies=False, other_smilies=False,
                      hugs_and_kisses=False,hearts=False,
                      hashtag=False, hashtag_mention=False, 
                      numbers=False, number_mention=False, 
                      exclamation=False,
                      set_to_not=False
                      
                      ):
    """
    Input:
    corpus: a corpus on the format as the output in creat_corpus
    all_smilies: if true, same effect as if pos_smilies, neg_smilies, and other_smilies were true
    pos_smilies: if true, positive smilies such as : ), : (, ; ), ( ;, :p, ;p, : p, are replaced by "possmiley"
    neg_smilies: if true, negative smilies such as : (, ) : are replaced by "negsmiely"
    other_smilies: if true, smilies such as ^_^ are replaced by a describing word. 
    hugs_and_kisses: if true, words such as xxx xoxo etc are replaced by "kisses" or "hug" and "kisses". 
    hearts: if true, "<3" are replaced by "heart"
    hashtags: if true, hashtags are removed from the beginning of words, so #apple becomes apple. 
    hashtag_mention: if true, and if hashtag is true, the word "hashatag" is added at the end of a tweet that used to contain
    one or more words beginning with a hashtag. 
    numbers: if true, words that are purely numbers are removed
    number_mention: if true, and if number is true, the word "thereisanumber" is added at the end of a tweet that used to contain
    one or more words that were purely numbers. 
    exclamation: if true, the word "exclamation" is added at the end of a tweet that contain one or more "!". 
    set_to_not: if true, all words ending with "n't" is replaced by not. 
   
    
    
    
    Output:
    new_corpus: a new corpus, on same format as the input corpus. 
    """
    #initialising the new corpus:
    new_corpus=[]

    #Want to split the tweets using this tokenizer:
    tknzr = TweetTokenizer(reduce_len=True)
    
    #Want to go though each line (tweet) in the corpus
    for line in corpus:
        
        
        if hashtag_mention:
            there_is_hashtag=False
        if number_mention:
            there_is_number=False
        if exclamation:
            there_is_exclamation=False
            
        #Splitting the tweet using the chosen tokenizer. 
        words=tknzr.tokenize(line)
        #Initializing for cleaned_tweet:
        cleaned_tweet=[]
        for i, word in enumerate(words):
            #Indicating that the word has not been treated yet
            word_not_treated=True
            if ((pos_smilies or all_smilies) and word_not_treated):
                if (i>0 and (word=='d' and (words[i-1]==':' or words[i-1]==';'))) or word == ':d' or word == ';d':
                    cleaned_tweet.append('possmiley')
                    word_not_treated=False
                elif (i>0 and (word=='p' and (words[i-1]==':' or words[i-1]==';'))) or word == ':p' or word == ';p' :
                    cleaned_tweet.append('possmiley')
                    word_not_treated=False
                elif i>0 and word=='d' and (words[i-1]==':' or words[i-1]==';' or words[i-1]=='x'):
                    cleaned_tweet.append('possmiley')
                    word_not_treated=False
                elif i>0 and words[i-1]=='(' and (word==':' or word==';'):
                    cleaned_tweet.append('possmiley')
                    word_not_treated=False
                elif i>0 and word==')' and (words[i-1]==':' or words[i-1]==';'):
                    cleaned_tweet.append('possmiley')
                    word_not_treated=False

            if ((neg_smilies or all_smilies) and word_not_treated):
                if i>0 and words[i-1]==')' and (word==':' or word==';'):
                    cleaned_tweet.append('negsmiley')
                    word_not_treated=False
                elif i>0 and word=='(' and (words[i-1]==':' or words[i-1]==';'):
                    cleaned_tweet.append('negsmiley')
                    word_not_treated=False
            
            if ((other_smilies or all_smilies) and word_not_treated):
                if i>0 and word=='_' and words[i-1]=='^' and words[i+1]=='^':
                    cleaned_tweet.append('eyesmiley')
                    word_not_treated=False
                elif i>0 and word=='o' and words[i-1]==':':
                    cleaned_tweet.append('openmouthface')
                    word_not_treated=False
                elif i>0 and word=='/' and words[i-1]==':':
                    cleaned_tweet.append('slashsmiely')
                    word_not_treated=False
                elif i>0 and word=='*' and (words[i-1]==':' or words[i-1]==';'):
                    cleaned_tweet.append('kiss')
                    word_not_treated=False
                
            if ((hugs_and_kisses and word_not_treated)):
                    #want to find hearts, hugs, kisses, etc: 
                if (word == "xoxo" or word == "xo" or word == "xoxoxo" or word == "xxoo"):
                    cleaned_tweet.append('hug')
                    cleaned_tweet.append('kiss')
                    word_not_treated=False
                elif (word=='xx' or word=='xxx'):
                    cleaned_tweet.append('kiss')
                    word_not_treated=False
            
            if ((hearts and word_not_treated)):
                if word == "<3":
                    cleaned_tweet.append('heart')
                    word_not_treated=False
            
            if (hashtag and word_not_treated):
                if word[0]=='#':
                    there_is_hashtag=True
                    cleaned_tweet.append(word[1:])
                    word_not_treated=False
            
            if (numbers and word_not_treated):
                if word.isdigit():
                    there_is_number=True
                    word_not_treated=False
                    
            if (exclamation and word_not_treated):
                if word=='!':
                    there_is_exclamation=True
                    cleaned_tweet.append(word)
                    word_not_treated=False
            
            if (set_to_not and word_not_treated):
                if word[-3:]=='n\'t':
                    cleaned_tweet.append('not')
                    word_not_treated=False
            
         
            if (word_not_treated and word!= '^' and word!=',' and word!='.' and word!=':' and word!='-' and word!='´'
                and word!=';'and word!=')' and word!='(' and word!='*'):
                if stemming:
                    cleaned_tweet.append(ps.stem(word))
                else:
                    cleaned_tweet.append(word)
                
        
        if (hashtag_mention and there_is_hashtag) :
            cleaned_tweet.append('hashtag')
        if (number_mention and there_is_number) :
            cleaned_tweet.append('thereisanumber')
        if (exclamation and there_is_exclamation):
            cleaned_tweet.append('exclamation')
            
            
        new_words = ' '.join(cleaned_tweet)
        new_words = new_words.encode('utf-8')
        new_corpus.append(new_words)
    return new_corpus       
    

