
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer


def indicate_number(corpus):
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
            words.append('detteerettallhallo')
       
        new_words = ' '.join(words)
        new_words = new_words.encode('utf-8')
        new_corpus.append(new_words) 

    
    
    return new_corpus

def corpus_to_strings(original_corpus):
    corpus=[]
    for line in original_corpus:
        words=line.decode("utf-8")#.split(" ")
        words = ''.join(words.rsplit('\n', 1))
        corpus.append(words)
    return corpus

def compare_set_char(neg_counter,pos_counter,characters):
    for char in characters:
        in_pos=int((pos_counter).get(char))
        in_neg=int((neg_counter).get(char))
        total=in_pos+in_neg
        print('percentage of ', char, 'in pos', in_pos/total, 'this is percentage in pos',in_pos/1250000)
        
def compare_sets(words,vec_pos,vec_neg):
    for word in words:
        in_pos=int(vec_pos.vocabulary_.get(word))
        in_neg=int(vec_neg.vocabulary_.get(word))
        total=in_pos+in_neg
        print('percentage of ', word, 'in pos', in_pos/total)
        
def strin_to_char(corpus):
    characters=[]
    for line in corpus[0:10]:
        for word in line:
            for char in word:
                characters.append(char)
    return characters

     