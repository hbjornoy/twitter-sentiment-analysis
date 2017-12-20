
#import random as rn
from keras import backend as K
#import tensorflow as tf
#from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
#import time
import sklearn as sk
import sklearn.preprocessing
from sklearn import model_selection
import numpy as np
import keras
#import neural_nets as NN
import tensorflow as tf
#import random as rn
from keras.models import load_model
#import re
#import os

import helpers as HL


""" 
TODO:
- Remove imports not used
"""

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

"""

def buildDocumentVector(tokens, size, model):

    vec = np.zeros(size).reshape((1, size))
    count = 0
    
    for word in tokens.split():
        try:
            word = word.decode('utf-8')
            word_vec = model[word].reshape((1, size))
            #idf_weighted_vec = word_vec * tfidf_dict[word]
            vec += word_vec
            count += 1
            
        except KeyError: # handling the case where the token is not in the corpus. useful for testing.
            if len(word.split('_')) > 1:
                word_vec = [0] * size
                p_count = 0
                
                for part in word.split('_'):
                    try:
                        part_vec = model[part].reshape((1, size))
                        if np.any(np.isinf(part_vec)):
                            print("part that contains inf", part)
                            print(part_vec[0:3])
                        word_vec += part_vec
                        p_count += 1
                        
                    except KeyError:
                        continue
                        
                if p_count != 0:
                    word_vec /= p_count
                vec += word_vec
                count += 1
            else:
                continue

    if count != 0:
        vec /= count
    return vec


def buildDocumentVector(tokens, size, model):

    vec = float(np.zeros(size).reshape((1, size)))
    count = 0
    
    for word in tokens.split():
        try:
            word = word.decode('utf-8')
            word_vec = model[word].reshape((1, size))
            #idf_weighted_vec = word_vec * tfidf_dict[word]
            vec += word_vec
            count += 1
            
        except KeyError: # handling the case where the token is not in the corpus. useful for testing.
            if len(word.split('_')) > 1:
                word_vec = [0] * size
                p_count = 0
                
                for part in word.split('_'):
                    try:
                        part_vec = model[part].reshape((1, size))
                        word_vec += part_vec
                        p_count += 1
                        
                    except:
                        continue
                        
                if p_count != 0:
                    word_vec /= p_count
                
                
                vec += word_vec
                count += 1
            else:
                continue

    if count != 0:
        vec /= count
    return vec
    
"""

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
    
    
def run_k_fold(models, X, Y, epochs, n_folds, patience):
    
    """
    Runs K-fold cross validation on the neural net models given as parameter models. 
    
    Input: 
        models: A list of names for neural net models defined in neural_nets.py
        X: Training set 
        Y: Labels 
        epochs: Max number of epochs to run each model in each k-fold
        n_folds: Number of folds for the k-fold algorithm
        
    Output:
        model_scores: A list of tuples containing mean and std for each model run 
            thorugh the k-fold cross validation.
    """
   
    #Need to set Keras-session in order to keep results reproducable
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    session = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(session)
    
    model_scores = []
    
    for neural_model in models:
 
        model_name = neural_model.__name__
        input_dimensions = X.shape[1]
        model = neural_model(input_dimensions)

        

        kfold = sk.model_selection.StratifiedKFold(n_splits=n_folds)
        
        cv_scores = []
       
        for train, test in kfold.split(X, Y):
            
            # Defining callbacks to be used under fitting process
            early_stopping= early_stopping_callback(patience_=patience, verbose_=1)


            model_checkpoint = model_checkpoint_callback("best_neural_model_save.hdf5", verbose_=1)
            
            model = neural_model(input_dimensions)
            
            model.fit(
                X[train], 
                Y[train], 
                epochs=epochs, 
                batch_size=1024, 
                verbose=1, 
                callbacks=[early_stopping, model_checkpoint], 
                validation_data=(X[test], Y[test])
            )
            
            #Load best model stored during fitting of model
            model = load_model('best_neural_model_save.hdf5')
            
            score = model.evaluate(X[test], Y[test], verbose=1)[1]
            cv_scores.append(score)
        
        model_mean = np.mean(cv_scores)
        model_std = np.std(cv_scores)
        
        print("Model: ", model_name)
        print("%.2f%% (+/- %.2f%%)" % (model_mean, model_std))
 
        model_scores.append([model_mean, model_std])
     
    return model_scores

def crossvalidation_for_dd(tuned_model, X, Y, epochs, n_folds, patience_):
    
    """
    
    TODO 
    
    """
    
    #Needed to keep results reproducable 
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    
    model_config = tuned_model.get_config()
        
    kfold = sk.model_selection.StratifiedKFold(n_splits=n_folds, shuffle=False)
    
    cv_scores = []
    histories = []
    
    for train, test in kfold.split(X, Y):
        
        #Defining callbacks to use during model fitting
        early_stopping_callback = early_stopping_callback(monitor='val_loss', patience=patience_, verbose=1)
        model_checkpoint_callback = model_checkpoint_callback("best_dd_model.hdf5", verbose=1)
        
        model = keras.models.Sequential.from_config(model_config)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(
            X[train],
            Y[train], 
            epochs=epochs, 
            batch_size=1024, 
            verbose=1, 
            callbacks=[early_stopping_callback, model_checkpoint_callback], 
            validation_data=(X[test], Y[test])
        )
        
        histories.append(history)
        
        score = model.evaluate(X[test], Y[test], verbose=0)
        cv_scores.append(score)
        
    print("Val_accuracies: %.2f%% (+/- %.4f)" % (np.mean([cv_score[1]*100 for cv_score in cv_scores]), 
                                               np.std([cv_score[1]*100 for cv_score in cv_scores])))     
    
    return model, cv_scores, histories
    

def testing_for_dd(tuned_model, X, Y, epochs, n_folds, patience_, split=0.9):
    
    split_size = int(X.shape[0]*split)
    
    # randomize
    np.random.seed(1337)
    shuffle_indexes = np.arange(X.shape[0])
    np.random.shuffle(shuffle_indexes)
    
    X = X[shuffle_indexes]
    Y = Y[shuffle_indexes]
    
    train_x, unseen_x = X[:split_size], X[split_size:]
    train_y, unseen_y = Y[:split_size], Y[split_size:]
    
    models = []
    cv_history = []
    histories = []
    
    start = time.time()
    model, cv_histories, histories = crossvalidation_for_dd(tuned_model, train_x, train_y , epochs, n_folds, patience_)
    
    train_score = model.evaluate(train_x, train_y)
    unseen_score = model.evaluate(unseen_x, unseen_y)
    print("evaluate on train_data: (loss:%.5f , acc:%.3f%%):" % (train_score[0], train_score[1]*100))
    print("Unseen_accuracies: (loss:%.4f , acc:%.4f%%):" % (unseen_score[0], unseen_score[1]*100))
    print("Time taken: ", (time.time() - start) / 60, "\n")
    
    return model, cv_histories, histories


def train_NN(model, allX, allY, patience_, epochs=100000, split=0.8):
    
    # Shuffling data in-place
    np.random.seed(1337)
    np.random.shuffle(allY)
    np.random.seed(1337)
    np.random.shuffle(allX)
    
    # defining the split index of the data
    split_size = int(allX.shape[0]*split)
    
     # Defining callbacks to be used under fitting process
    early_stopping_callback = early_stopping_callback(monitor='val_loss', patience=patience_, verbose=1)
    model_checkpoint_callback = model_checkpoint_callback("train_NN_dynamic_model.hdf5", verbose=1)
    
    history = []
    start = time.time()
    
    try:
        history = model.fit(
            allX[:split_size], 
            allY[:split_size], 
            epochs=epochs, 
            batch_size=1024, 
            verbose=1,
            callbacks=[early_stopping_callback, model_checkpoint_callback], 
            validation_data=(allX[split_size:], allY[split_size:])
        )
        
        return model, history
        
    except (KeyboardInterrupt, SystemExit):
        print('\n\ntime spent training:', (time.time() - start))
        return model, history
    else:
        print("\n\nwhy did this happen?")
        print(model, history)
        return model, history

    
    
def classify_with_neural_networks(neural_nets_functions, global_vectors, processed_corpus, total_training_tweets, nr_pos_tweets, epochs, n_folds,patience=3):
 
    num_of_dim = global_vectors.syn0.shape[1]
 
    # seperate traindata and testdata
    train_corpus = processed_corpus[:total_training_tweets:]
    predict_corpus = processed_corpus[total_training_tweets::]
 
    # Build a vector of all the words in a tweet
    vectors = np.zeros(len(train_corpus), dtype=object)
    for i, doc in enumerate(train_corpus):
        if (i % 50000) == 0:
            print("tweets processed: %.0f  of total number of tweets: %.0f" % (i,len(train_corpus)))
        vectors[i] = buildDocumentVector(doc, num_of_dim, global_vectors)
    
    train_document_vecs = np.concatenate(vectors)
    train_document_vecs = sklearn.preprocessing.scale(train_document_vecs)

    labels = HL.create_labels(nr_pos_tweets, nr_pos_tweets)
 
    model_scores= run_k_fold(neural_nets_functions, train_document_vecs, labels, epochs, n_folds,patience)
    
    return model_scores


def shuffle_data(X, Y):
    np.random.seed(1337)
    np.random.shuffle(X)
    np.random.seed(1337)
    np.random.shuffle(Y)
   
    return X, Y
   
    
def split_data(X, Y, split=0.8):
    split_size = int(X.shape[0]*split)
    train_x, val_x = X[:split_size], X[split_size:]
    train_y, val_y = Y[:split_size], Y[split_size:]
   
    return train_x, val_x, train_y, val_y

def model_checkpoint_callback(save_filename, verbose_,):
    
    return keras.callbacks.ModelCheckpoint(
        "best_neural_model_save.hdf5",
        monitor='val_loss',
        verbose=verbose_, 
        save_best_only=True,
        save_weights_only=False, 
        mode='auto'
    )

def early_stopping_callback(patience_, verbose_):
    
    return keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_, verbose = verbose_)
    
def get_prediction(neural_net, global_vectors, full_corpus, total_training_tweets, nr_pos_tweets,kaggle_name, epochs, patience, split=0.8):
    """ Creates a csv file with kaggle predictions and returns the predictions.
    Input:
        neural_net: Name of a neural net model
        global_vectors: global vectors created out the gensim-.txt files.
        total_training_tweets: (int) Number of tweets that are training tweets. Assums that the first poriton of the corpus is
        training tweets, the second part is the unseen test set.
        nr_pos_tweets: (int) number of traning tweets that are positiv
        kaggle_name: Name for csv file, must end in '.csv'.
   
    Output:
        pred_ones: the predicions (1 or -1)
        a .csv file with name 'kaggle_name'
    """
    num_of_dim = global_vectors.syn0.shape[1]
    
    # seperate traindata and testdata
    train_corpus = full_corpus[:total_training_tweets:]
    predict_corpus = full_corpus[total_training_tweets::]
    
    # Build a vector of all the words in a tweet
    train_document_vecs = np.concatenate([buildDocumentVector(doc, num_of_dim, global_vectors) for doc in train_corpus])
    train_document_vecs = sk.preprocessing.scale(train_document_vecs)
    
    labels = HL.create_labels(nr_pos_tweets, nr_pos_tweets, kaggle=True)
       
    train_document_vecs, labels = shuffle_data(train_document_vecs,labels)
    train_x, val_x, train_y, val_y = split_data(train_document_vecs, labels, split)
       
    test_document_vecs = np.concatenate([buildDocumentVector(doc, num_of_dim, global_vectors) for doc in predict_corpus])
    test_document_vecs = sk.preprocessing.scale(test_document_vecs)
   
    model_name = neural_net.__name__
    model = neural_net(num_of_dim)
    
    # Defining callbacks to be used under fitting process
    early_stopping = early_stopping_callback(patience_=patience, verbose_=1)
    model_checkpoint = model_checkpoint_callback("best_neural_model_prediction_model.hdf5", verbose_=1)
    
    history = model.fit(
        train_x,
        train_y,
        epochs=epochs,
        batch_size=1024,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint],
        validation_data=(val_x, val_y)
    )
   
    # Loading the best model found during training
    model = load_model('best_neural_model_prediction_model.hdf5')
 
    prediction = model.predict(test_document_vecs)
   
    prediction = [1 if i > 0.5 else -1 for i in prediction]
           
    #CREATING SUBMISSION
    ids = list(range(1,10000+1))
    HL.create_csv_submission(ids, prediction,kaggle_name)
 
    return prediction