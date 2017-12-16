import numpy as np
import nltk
import enchant
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer

from nltk.tokenize import TweetTokenizer

from ekphrasis.classes.segmenter import Segmenter
from ekphrasis.classes.spellcorrect import SpellCorrector
def preprocess_corpus(corpus,stemming=False,
                      all_smilies=False, pos_smilies=False, neg_smilies=False, other_smilies=False,
                      hugs_and_kisses=False,hearts=False,
                      hashtag=False, hashtag_mention=False, 
                      numbers=False, number_mention=False, 
                      exclamation=False,  ##OBS denne er nå ikke testet, eventuelt bare fjerne den
                      set_to_not=False, 
                      segmentation_hash= False, 
                      spelling=False,
                      elongation=False, 
                      remove_signs=False
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
    segmentation_hash: if true, words starting with # that do not appear in the english dictionary is split into segments, 
    eg '#iammoving' becomes 'i am moving'
    spelling: if true, all words that are not a part of the english dictionary is set to the most likely word,
    within two alterations
    elongation: if true, the length of all sequences of letters in words that are not a part of the English dictionary 
    is set to max 2. Before words that are altered because of this, the word 'elongation' appears. 
    remove_signs: if true, signs such as ",", ".", ":", ";", "-", are removed. 
    
    Output:
    new_corpus: a new corpus, on same format as the input corpus. 
    """
   
    start = time.time()
    
    #initialising the new corpus:
    new_corpus=[]

    #Want to split the tweets using this tokenizer:
    tknzr = TweetTokenizer(reduce_len=True)
    
    
    
    if stemming:
        ps = PorterStemmer()
    
    if segmentation_hash or spelling or elongation:
        d = enchant.Dict("en_US")
    
    if segmentation_hash: 
        #seg = Segmenter(corpus="english")
        seg = Segmenter(corpus="twitter")

    if spelling: 
        sp = SpellCorrector(corpus="english")
        
    
    elapsed = time.time()
    print("Time in min before starting first for loop:", (elapsed - start) / 60 )
    
    #Want to go though each line (tweet) in the corpus
    for k, line in enumerate(corpus):
        
        
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
            end_=len(words)-1
            if ((pos_smilies or all_smilies) and word_not_treated):
                if (i>0 and (word=='d' and (words[i-1]==':' or words[i-1]==';'))) or word == ':d' or word == ';d':
                    cleaned_tweet.append('smile')
                    word_not_treated=False
                elif (i>0 and (word=='p' and (words[i-1]==':' or words[i-1]==';'))) or word == ':p' or word == ';p' :
                    cleaned_tweet.append('smile')
                    word_not_treated=False
                elif i>0 and word=='d' and (words[i-1]==':' or words[i-1]==';' or words[i-1]=='x'):
                    cleaned_tweet.append('smile')
                    word_not_treated=False
                elif i>0 and words[i-1]=='(' and (word==':' or word==';'):
                    cleaned_tweet.append('smile')
                    word_not_treated=False
                elif i>0 and word==')' and (words[i-1]==':' or words[i-1]==';'):
                    cleaned_tweet.append('smile')
                    word_not_treated=False

            if ((neg_smilies or all_smilies) and word_not_treated):
                if i>0 and words[i-1]==')' and (word==':' or word==';'):
                    cleaned_tweet.append('sad')
                    word_not_treated=False
                elif i>0 and word=='(' and (words[i-1]==':' or words[i-1]==';'):
                    cleaned_tweet.append('sad')
                    word_not_treated=False
            
            if ((other_smilies or all_smilies) and word_not_treated):
                if i>0  and i<end_ and word=='_' and words[i-1]=='^' and words[i+1]=='^':
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
                elif (word=='xx' or word=='xxx'or word=='xxxx'):
                    cleaned_tweet.append('kiss')
                    word_not_treated=False
            
            if ((hearts and word_not_treated)):
                if word == "<3":
                    cleaned_tweet.append('heart')
                    word_not_treated=False
            
            if (hashtag and word_not_treated):
                if word[0]=='#':
                    there_is_hashtag=True
                    if (len(word)>1 and segmentation_hash and not d.check(word[1:])):
                        cleaned_tweet.append(seg.segment(word[1:]))
                    else:
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
           
            
         
            if (word_not_treated):
                if (not remove_signs) or (remove_signs and ( (word!= '^' and word!=',' and word!='.' and word!=':' 
                                                              and word!='-' and word!='´' and word!=';'and word!=')' 
                                                              and word!='(' and word!='*'))):
                  
                    if ((not word[0].isdigit()) and elongation and not d.check(word) and len(word)>2):
                        new=[]
                        new.append(word[0])
                        for i,letter in enumerate(word):
                            if i>0 and i<len(word)-1: 
                                if not( letter==word[i-1]==word[i+1]):
                                    new.append(letter)
                        new.append(word[-1])
                        new_word=''.join(new)
                        if new_word!= word:
                            cleaned_tweet.append('elongation')
                            word=new_word

                    if spelling and not d.check(word)and len(word)>2: 
                        word=sp.correct(word)
                    if stemming:
                        word=ps.stem(word)

                    
                    cleaned_tweet.append(word)

           
                
        
        if (hashtag_mention and there_is_hashtag) :
            cleaned_tweet.append('hashtag')
        if (number_mention and there_is_number) :
            cleaned_tweet.append('number')
        if (exclamation and there_is_exclamation):
            cleaned_tweet.append('exclamation')
            
            
        new_words = ' '.join(cleaned_tweet)
        new_words = new_words.encode('utf-8')
        new_corpus.append(new_words)
        
        if np.mod(k,25000)==1:
                elapsed = time.time()
                print("Time in min after", k, " tweets:", (elapsed - start) / 60 )

        
    elapsed = time.time()
    print("Time in min total:", (elapsed - start) / 60 )
    return new_corpus       
    

def get_dynamic_stopwords(corpus, MinDf, MaxDf,sublinearTF=True,useIDF=False):
    """
    Input: 
    corpus: A corpus, 
    MinDf: as min_df in sklearns TfidVectorizer:
    MaxDf: as max_df in sklearns TfidVectorizer:
    sublinearTF: as sublinear_tf in sklearns TfidVectorizer:
    useIDF: as use_idf in sklearns TfidVectorizer:
    
    Output: 
    vectorizer.stop_words_: A list of all words that should be removed form the corpus. 
    """
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
    Input: 
    corpus: A corpus 
    custom_stop_words: A list of words that should be removed
    
    outout: 
    new_corpus: A new corpus, as 'corpus', but all words in custom_stop_words are removed
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

