

import numpy as np

import pickle





def make_file(embedding_npy_file,vocab_pkl_file, textfile_name):
    """
    
    MULIG JEG GJØR EN FEIL I DENNE, MEN TROR EGENTLIG IKKE DET (HVORDAN MATCHE ORD TIL VEKTOR)¨
    
    Input: 
    embedding_npy_file: A string, which is the disired name for your embedding file, not containing the ending .npy. 
    If you want to name your file 'embeddings.npy', then embedding_npy_file='embeddings'. 
    vocab_pkl_file: A string, which is the name of your vocab pickle file, eg vocab.pkl
    textfile_name: A string, which is the desired name of the textfile storing the word vectors, eg. 'global_vectors.txt'. 
    
    
    Output: A file, named textfile_name, in which each line is a word, then its vector. 
    
    """
    
    global_vectors=np.load(embedding_npy_file)
    
    with open(vocab_pkl_file, 'rb') as f:
        vocab = pickle.load(f)
    
    dim=len(global_vectors[0])
    
    
    with open(textfile_name, 'w') as f:

        f.write(str(len(vocab)))
        f.write(' ')
        f.write(str(dim))
        f.write('\n')
        for  word in (vocab):
            f.write(word)
            f.write(' ')
            index=vocab.get(word,-1)
            if index >= 0:
                for number in global_vectors[index]:
                    f.write(str(number))
                    f.write(' ')
                f.write('\n')

def corpus_to_textfile(corpus,filename):
    
    """
    Input 
    corpus: A corpus: List of binary strings, each binary string contains a tweet. 
    filename: A string, the desired name of the text file that will contain the corpus, eg 'corpus.txt'. 
    
    Output: A file, named filename, which containts the corpus in the same way the original data files contains the 
    unpreprocessed corpus. 
    """
    
    with open(filename, 'w') as f: 
        for line in corpus:
            line=line.decode("utf-8")
            f.write(line)
            f.write('\n')
            
#def vocab_from_corpusfile(vocab_filename,corpus_file): 
    
    