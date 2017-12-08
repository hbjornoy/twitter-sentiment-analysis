from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
import time
import sklearn as sk
import numpy as np
import keras
import neural_nets as NN
import tensorflow as tf
import random as rn
from keras import backend as K

import helpers as HL 
np.random.seed(7)
rn.seed(7)
tf.set_random_seed(7)


def create_gensim_word2vec_file(path_to_original_glove_file):
    """
    It creates a gensim_glove_vectors.txt-file in the same folder, that can be downloaded after

    :param path_to_original_glove_file: This the relative path to the original pretrained Glovefile must be given.
    """
    # spits out a .txt-file with the vectors in gensim format
    glove2word2vec(glove_input_file=path_to_original_glove_file, word2vec_output_file="gensim_global_vectors.txt")


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
    count = 0.
    for word in tokens.split():
        try:
            word = word.decode('utf-8')
            word_vec = model[word].reshape((1, size))
            #idf_weighted_vec = word_vec * tfidf_dict[word]
            vec += word_vec
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


def create_labels(total_training_tweets, nr_pos_tweets):
    """
    THIS IS BAD CODE AND SHOULD BE FIXED:
    should just get the labels directly..
    """
    # Making labels
    labels = np.zeros(total_training_tweets)
    labels[0:nr_pos_tweets] = 1
    labels[nr_pos_tweets:total_training_tweets] = 0
    return labels


def run_k_fold(models, X, Y, epochs, n_folds, seed):
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    for neural_model in models:

        model_name = neural_model.__name__

        input_dimensions = X.shape[1]
        model = neural_model(input_dimensions)

        start = time.time()

        kfold = sk.model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        cv_scores = []
        pos_scores = []
        neg_scores = []
        ratio_of_pos_guesses = []

        for train, test in kfold.split(X, Y):
            early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

            model.fit(X[train], Y[train], epochs=epochs, batch_size=1024, verbose=0, callbacks=[early_stopping])

            score = model.evaluate(X[test], Y[test], verbose=0)
            cv_scores.append(score[1] * 100)

            # To analyze if it is unbalanced classifying
            labels = Y[test]
            pred = model.predict(X[test])
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
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
        print("Negative sentiment: %.2f%%  Positive sentiment: %.2f%%" % (np.mean(neg_scores), np.mean(pos_scores)))
        print("Percentage of positive classifications (should be 50%ish):", np.mean(ratio_of_pos_guesses)*100)
        print("Time taken: ", (time.time() - start) / 60, "\n")


def classify_with_neural_networks(neural_nets_functions, global_vectors, processed_corpus, total_training_tweets, nr_pos_tweets):

    num_of_dim = global_vectors.syn0.shape[1]

    # seperate traindata and testdata
    train_corpus = processed_corpus[:total_training_tweets:] 
    predict_corpus = processed_corpus[total_training_tweets::] 

    # Build a vector of all the words in a tweet
    train_document_vecs = np.concatenate([buildWordVector(doc, num_of_dim, global_vectors) for doc in train_corpus])
    train_document_vecs = sk.preprocessing.scale(train_document_vecs)

    labels = create_labels(total_training_tweets, nr_pos_tweets)

    run_k_fold(neural_nets_functions, train_document_vecs, labels, epochs=10, n_folds=3, seed=7)

def method1(path_to_gensim_global_vectors, processed_corpus, total_training_tweets, nr_pos_tweets, all_neural_nets=False):

    global_vectors = make_glove(path_to_gensim_global_vectors)

    if all_neural_nets == False:
        neural_nets = [NN.deep_HB]
    elif all_neural_nets == True:
        neural_nets = [NN.basic_model, NN.basic_model_adam, NN.wide_model, NN.deep_2_model, NN.deep_HB]
    else:
        neural_nets = all_neural_nets

    classify_with_neural_networks(neural_nets, global_vectors, processed_corpus, total_training_tweets, nr_pos_tweets)



def get_prediction(neural_net, global_vectors, full_corpus, total_training_tweets, nr_pos_tweets,kaggle_name):
    num_of_dim = global_vectors.syn0.shape[1]
    # seperate traindata and testdata
    train_corpus = full_corpus[:total_training_tweets:] 
    predict_corpus = full_corpus[total_training_tweets::] 
    # Build a vector of all the words in a tweet
    train_document_vecs = np.concatenate([buildWordVector(doc, num_of_dim, global_vectors) for doc in train_corpus])
    train_document_vecs = sk.preprocessing.scale(train_document_vecs)
    
    test_document_vecs = np.concatenate([buildWordVector(doc, num_of_dim, global_vectors) for doc in predict_corpus])
    test_document_vecs = sk.preprocessing.scale(test_document_vecs)

    labels = create_labels(total_training_tweets, nr_pos_tweets)
    
    model_name = neural_net.__name__
    
    model = neural_net(num_of_dim)
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    model.fit(train_document_vecs, labels, epochs=10, batch_size=1024, verbose=0, callbacks=[early_stopping])
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


