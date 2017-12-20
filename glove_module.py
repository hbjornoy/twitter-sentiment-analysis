
import random as rn
from keras import backend as K
import tensorflow as tf

from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
import time
import sklearn as sk
import sklearn.preprocessing
from sklearn import model_selection
import numpy as np
import keras
import neural_nets as NN
import tensorflow as tf
import random as rn
from keras import backend as K
from keras.models import load_model
import re
import os
import seaborn as sb

import helpers as HL

from numpy.random import seed


def create_gensim_word2vec_file(path_to_original_glove_file):
    """
    :param path_to_glove_folder: Go to Stanfords website https://nlp.stanford.edu/projects/glove/ and download their twitterdataset,
            put it in the same folder as this function and write the path to it as the input of this function
    :return: nothing but creates a .txt-file with the gensim object, afterwards you can delete the original glove-files
            and keep the created gensim___.txt files and use the function load_gensim_global_vectors(path_to_global_vectors) to load them in.
    """

    for filename in os.listdir(path_to_original_glove_file):
        if filename.endswith(".txt"):
            filepath = os.path.join(path_to_original_glove_file, filename)
            dim = get_dim_of_file(filename)
            name_of_filecreated = "gensim_global_vectors_"+ dim + "dim.txt"

            # spits out a .txt-file with the vectors in gensim format
            glove2word2vec(glove_input_file=filepath, word2vec_output_file="gensim_global_vectors_"+ dim + "dim.txt")
            print(name_of_filecreated)
            continue
        else:
            continue

def get_dim_of_file(filename):
    removed_27 = filename.replace("27", "")
    # after removing 27 it should only be dim_nr. that is numerical, create regex that filter nonnumericals out
    non_decimal = re.compile(r'[^\d]+')
    dim = non_decimal.sub('', removed_27)
    return dim


def make_glove(path_to_gensim_global_vectors):
    """
    uses the created gensim-.txt file to create the word2vec so one can operate on

    :param corpus: Corpus is the set of documents you want to train on after preprocessing
    :param path_to_pretrained_glove:
    :return: global vectors in the form of a list of words, with the words represented as vectors
            with length the same as the dimensions of the space
    """
    glove_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_gensim_global_vectors, binary=False)

    global_vectors = glove_model.wv
    del glove_model

    return global_vectors


def buildWordVector(tokens, size, model):

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


def create_labels(total_training_tweets, nr_pos_tweets):
    
    """
    THIS IS BAD CODE AND SHOULD BE FIXED:
    should just get the labels directly..
    TROR IKKE VI BRUKER DENNE?
    """
    # Making labels
    labels = np.zeros(total_training_tweets)
    labels[0:nr_pos_tweets] = 1
    labels[nr_pos_tweets:total_training_tweets] = 0
    return labels


def run_k_fold(models, X, Y, epochs, n_folds):
   
    #Needed to keep results reproducable
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    
    seed(1337)
    
    s = np.arange(X.shape[0])
    
    X = X[s]
    Y = Y[s]
    
    #unseen_x = X[:10000]
    #unseen_y = Y[:10000]

    x_test = X[:40000]
    y_test = Y[:40000]
   
    X = X[40000:]
    Y = Y[40000:] 
    
    model_scores = []
    for neural_model in models:
 
        model_name = neural_model.__name__
       
        input_dimensions = X.shape[1]
        model = neural_model(input_dimensions)
        start = time.time()

        kfold = sk.model_selection.StratifiedKFold(n_splits=n_folds)
        cv_scores = []
        pos_scores = []
        neg_scores = []
        ratio_of_pos_guesses = []
       
        for train, test in kfold.split(X, Y):
            
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)
            
            model_checkpoint = keras.callbacks.ModelCheckpoint("best_neural_model_save.hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
            
            model = neural_model(input_dimensions)
            
            history = model.fit(X[train], Y[train], epochs=epochs, batch_size=1024, verbose=1, callbacks=[early_stopping, model_checkpoint], validation_data=(x_test, y_test))
            
            model = load_model('best_neural_model_save.hdf5')
            
            score = model.evaluate(X[test], Y[test], verbose=1)[1]
            
            print("Unseen score:", score)

            pred = model.predict(X[test])
            
            cv_scores.append(score)
 
            # To analyze if it is unbalanced classifying (SLETTE ALT DETTE)
            labels = Y[test]
            pos_right = 0
            neg_right = 0
            for i, label in enumerate(labels):
                if label == 1 and pred[i] >=0.5:
                    pos_right += 1
                elif label == 0 and pred[i] < 0.5:
                    neg_right += 1
            ratio_of_pos_guesses.append(np.mean(np.round(np.array(pred)).astype(int)))
            pos_scores.append((pos_right / (len(labels) * 0.5))*100)
            neg_scores.append((neg_right / (len(labels) * 0.5))*100)
 
        print("Model: ", model_name)
        print(history)
        print(cv_scores)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
        print("Negative sentiment: %.2f%%  Positive sentiment: %.2f%%" % (np.mean(neg_scores), np.mean(pos_scores)))
        print("Percentage of positive classifications (should be 50%ish):", np.mean(ratio_of_pos_guesses)*100)
        print("Time taken: ", (time.time() - start) / 60, "\n")
 
        model_scores.append((np.mean(cv_scores), np.std(cv_scores)))
     
    return model_scores

def crossvalidation_for_dd(tuned_model, X, Y, epochs, n_folds):
    #Needed to keep results reproducable 
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    
    model_config = tuned_model.get_config()
    
    start = time.time()
    kfold = sk.model_selection.StratifiedKFold(n_splits=n_folds, shuffle=False)
    cv_scores = []
    histories = []
    pos_scores = []
    neg_scores = []
    ratio_of_pos_guesses = []
    
    for train, test in kfold.split(X, Y):
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        model_checkpoint = keras.callbacks.ModelCheckpoint("best_dd_model.hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
        
        model = keras.models.Sequential.from_config(model_config)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(X[train], Y[train], epochs=epochs, batch_size=1024, verbose=1, callbacks=[early_stopping, model_checkpoint], validation_data=(X[test], Y[test]))
        score = model.evaluate(X[test], Y[test], verbose=0)
        cv_scores.append(score) # end results of the cv
        histories.append(history)
        
    print("Val_accuracies: %.2f%% (+/- %.4f)" % (np.mean([cv_score[1]*100 for cv_score in cv_scores]), 
                                               np.std([cv_score[1]*100 for cv_score in cv_scores])))
    print("Time taken: ", (time.time() - start) / 60, "\n")
            
    return model, cv_scores, histories
    

def testing_for_dd(tuned_model, X, Y, epochs, n_folds, split=0.9):
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
    model, cv_histories, histories = crossvalidation_for_dd(tuned_model, train_x, train_y , epochs, n_folds)
    
    train_score = model.evaluate(train_x, train_y)
    unseen_score = model.evaluate(unseen_x, unseen_y)
    print("evaluate on train_data: (loss:%.5f , acc:%.3f%%):" % (train_score[0], train_score[1]*100))
    print("Unseen_accuracies: (loss:%.4f , acc:%.4f%%):" % (unseen_score[0], unseen_score[1]*100))
    print("Time taken: ", (time.time() - start) / 60, "\n")
    
    return model, cv_histories, histories

def train_NN(model, allX, allY, epochs=100000, split=0.8):
    # Shuffling data in-place
    np.random.seed(1337)
    np.random.shuffle(allY)
    np.random.seed(1337)
    np.random.shuffle(allX)
    
    # defining the split index of the data
    split_size = int(allX.shape[0]*split)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    model_checkpoint = keras.callbacks.ModelCheckpoint("train_NN_dynamic_model.hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    
    history = []
    start = time.time()
    try:
        history = model.fit(allX[:split_size], allY[:split_size], epochs=epochs, batch_size=1024, verbose=1, callbacks=[early_stopping, model_checkpoint], validation_data=(allX[split_size:], allY[split_size:]))
        
        return model, history
        
    except (KeyboardInterrupt, SystemExit):
        print('\n\ntime spent training:', (time.time() - start))
        return model, history
    else:
        print("\n\nwhy did this happen?")
        print(model, history)
        return model, history

def plot_history(history):
    """should make this to plot the history of epochs and validationscore
    maybe even the crossvalidation mean of at each epoch? smoothen out the graph :)
    BRUKER VI DENNE? I SÅ FALL MÅ VI SKRIVE OM SEABORN I READ ME
    - make history into dataframe that fits seaborn
    - epoch on the x axis
    - score on the y axix (0-1)
    - plot val_los, val_acc, train_acc and train_loss
    """
    
    import seaborn as sns
    sb.set(style="darkgrid")

    # Load the long-form example gammas dataset
    gammas = sns.load_dataset("gammas")

    # Plot the response with standard error
    sb.tsplot(data=gammas, time="timepoint", unit="subject",
           condition="ROI", value="BOLD signal")
    
    
    
def classify_with_neural_networks(neural_nets_functions, global_vectors, processed_corpus, total_training_tweets, nr_pos_tweets, epochs, n_folds):
 
    num_of_dim = global_vectors.syn0.shape[1]
 
    # seperate traindata and testdata
    train_corpus = processed_corpus[:total_training_tweets:]
    predict_corpus = processed_corpus[total_training_tweets::]
 
    # Build a vector of all the words in a tweet
    vectors = np.zeros(len(train_corpus), dtype=object)
    for i, doc in enumerate(train_corpus):
        if (i % 50000) == 0:
            print("tweets processed: %.0f  of total number of tweets: %.0f" % (i,len(train_corpus)))
        vectors[i] = buildWordVector(doc, num_of_dim, global_vectors)
    train_document_vecs = np.concatenate(vectors)
    train_document_vecs = sklearn.preprocessing.scale(train_document_vecs)

    labels = create_labels(total_training_tweets, nr_pos_tweets)
 
    model_scores= run_k_fold(neural_nets_functions, train_document_vecs, labels, epochs, n_folds)
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


def get_prediction(neural_net, global_vectors, full_corpus, total_training_tweets, nr_pos_tweets,kaggle_name, epochs, split=0.8):
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
    train_document_vecs = np.concatenate([buildWordVector(doc, num_of_dim, global_vectors) for doc in train_corpus])
    train_document_vecs = sk.preprocessing.scale(train_document_vecs)
    labels = create_labels(total_training_tweets, nr_pos_tweets)
   
    train_document_vecs, labels = shuffle_data(train_document_vecs,labels)
    train_x, val_x, train_y, val_y = split_data(train_document_vecs, labels, split)
   
    test_document_vecs = np.concatenate([buildWordVector(doc, num_of_dim, global_vectors) for doc in predict_corpus])
    test_document_vecs = sk.preprocessing.scale(test_document_vecs)
   
    model_name = neural_net.__name__
   
    model = neural_net(num_of_dim)
   
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = keras.callbacks.ModelCheckpoint("best_neural_model_prediction_model.hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
 
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=1024, verbose=1, callbacks=[early_stopping, model_checkpoint], validation_data=(val_x, val_y))
   
    model = load_model('best_neural_model_prediction_model.hdf5')
 
    pred=model.predict(test_document_vecs)
   
    pred_ones=[]
    for i in pred:
        if i> 0.5:
            pred_ones.append(1)
        else:
            pred_ones.append(-1)
           
    #CREATING SUBMISSION
    ids = list(range(1,10000+1))
    HL.create_csv_submission(ids, pred_ones,kaggle_name)
 
    return pred_ones


