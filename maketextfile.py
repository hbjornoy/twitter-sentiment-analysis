

import numpy as np

import pickle


#import files given by the ML course: 

import cooc as CO
import pickle_vocab as PV

def make_file(embedding_npy_file,vocab_pkl_file, textfile_name):
    
    global_vectors=np.load(embedding_npy_file)
    
    with open(vocab_pkl_file, 'rb') as f:
        vocab = pickle.load(f)
    
    dim=len(global_vectors[0])
    
    
    with open(textfile_name, 'w') as f:

        f.write(str(len(vocab)))
        f.write(' ')
        f.write(str(dim))
        f.write('\n')
        for i, word in enumerate(vocab):
            f.write(word)
            f.write(' ')
            for number in global_vectors[i]:
                f.write(str(number))
                f.write(' ')
            f.write('\n')

def corpus_to_textfile(corpus,filename):
    with open(filename, 'w') as f: 
        for line in corpus[0:10]:
            line=line.decode("utf-8")
            f.write(line)
            f.write('\n')
            
def vocab_from_corpusfile(vocab_filename,corpus_file): 
    
    