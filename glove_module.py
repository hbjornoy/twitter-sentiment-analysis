from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
import numpy as np
import re
import os


from numpy.random import seed

def create_gensim_word2vec_file(path_to_original_glove_file):
    
    """
    Creates gensim_global_vectors_<dim> files from in a format readable by the gensim.models.KeyedVectors.load_word2vec_format method. 
    
    Input:
        path_to_original_glove_file: Path to the glove files downloaded from the 
    """
    
    for filename in os.listdir(path_to_original_glove_file):
        
        if filename.endswith(".txt"):
            
            filepath = os.path.join(path_to_original_glove_file, filename)
            dim = get_dim_of_file(filename)
            name_of_filecreated = "gensim_global_vectors_"+ dim + "dim.txt"

            # Creates a .txt-file with the vectors in gensim format
            glove2word2vec(glove_input_file=filepath, word2vec_output_file="gensim_global_vectors_"+ dim + "dim.txt")

            
def get_dim_of_file(filename):
    
    """
    Helper method to find sub_string in filename representing the dimention of the glove-vector
    
    Input: 
        Filename: The name of the file in which to find the dimension

    Output:
        dim: The dimension of the file
    """
    
    removed_27 = filename.replace("27", "")
    non_decimal = re.compile(r'[^\d]+')
    dim = non_decimal.sub('', removed_27)
    return dim


def create_glove_model(path_to_gensim_global_vectors):
    """
    Uses the created gensim-.txt file to create the word2vec so one can operate on
    
    Input:
        path_to_gensim_global_vectors: Corpus is the set of documents you want to train on after preprocessing
        
    Output:
        global_vectors: Returns a gensim word embedding object containing the glove-embeddings.
    """
    
    glove_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_gensim_global_vectors, binary=False)

    global_vectors = glove_model.wv
    del glove_model

    return global_vectors


def buildDocumentVector(document, vec_dimention, word_embedding_model):
    
    """
    Builds a vector representation of each document(tweet) of the given dimention, by finding the mean of all word vectors.
    
    Input:
        document: A tweet in string form
        vec_dimention: The dimention of vector used to represent words embeddings
        word_embedding_model: The word embedding model used
        
    Output:
        A vector representing a tweet, using the mean of the word embeddings. 
    """
    
    document_vec = np.zeros(vec_dimention).reshape((1, vec_dimention))
    count = 0
    
    for word in document.split():
        try:
            word = word.decode('utf-8')
            word_vec = word_embedding_model[word].reshape((1, vec_dimention))
            document_vec += word_vec
            count += 1
            
        except KeyError: 
            
            # If the word is an n_gram, represent the n_gram as the mean of all sub_words in the n_gram. 
            if len(word.split('_')) > 1:
                word_vec = build_word_vec_for_n_gram(word, vec_dimention, word_embedding_model)
                document_vec += word_vec
                count += 1

    #Finding mean of all word vectors in the document vector
    if count != 0:
        document_vec /= count
        
    return document_vec


def build_word_vec_for_n_gram(n_gram, vec_dimention, word_embedding_model):
    
    """
    Builds a vector representation for an n_gram of the given dimention, 
           by finding the mean of all word vectors.
    Input:
        n_gram: An n_gram in string form
        vec_dimention: The dimention of vector used to represent words embeddings
        word_embedding_model: The word embedding model used
   
    Output:
        A vector representing an n_gram, using the mean of the sub_word embeddings. 
    """
    word_vec = [0] * vec_dimention
    partial_count = 0
    try: 
        for part in n_gram.split('_'):
            try:
                part_vec = word_embedding_model[part].reshape((1, vec_dimention))
                word_vec += part_vec
                partial_count += 1

            except KeyError:
                continue

        if partial_count != 0:
            word_vec /= partial_count
            return word_vec

        return word_vec
    
    except:
        return None
    