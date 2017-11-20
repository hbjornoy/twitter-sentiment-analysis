import os
from gensim import corpora
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

def stopAndStem(inputfile, outputfile):
    #loading the data file: 
    with open(inputfile) as f:
        documents = [ line.strip( ) for line in list(f) ]
    #Defining stoplist:   
    stoplist = set('for a of the and to in is i <user> <url> !'.split())
    #removing stopwords: 
    texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]
    #stemming:
    texts=[[ps.stem(word) for word in sentence ]for sentence in texts]
    file = open(outputfile,'w')  
 
    for i in range(len(texts)):
        file.write(' '.join(texts[i]) + '\n' )
    
    file.close()