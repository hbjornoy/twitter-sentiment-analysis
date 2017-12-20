from keras import backend as K
import tensorflow as tf
import sklearn as sk
import sklearn.preprocessing
from sklearn import model_selection
import numpy as np
import keras
import tensorflow as tf
from keras.models import load_model

import helpers as HL
import glove_module as GM

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
 
        model_scores.append(model_mean, model_std)
     
    return model_scores


    

def classify_with_neural_networks(neural_nets_functions, global_vectors, processed_corpus, total_training_tweets, nr_pos_tweets, epochs, n_folds,patience=3):
    
    """
    
    """
 
    num_of_dim = global_vectors.syn0.shape[1]
 
    # seperate traindata and testdata
    train_corpus = processed_corpus[:total_training_tweets:]
    predict_corpus = processed_corpus[total_training_tweets::]
 
    # Build a vector representation of all documents in corpus
    vectors = np.zeros(len(train_corpus), dtype=object)
    for i, doc in enumerate(train_corpus):
        if (i % 50000) == 0:
            print("tweets processed: %.0f  of total number of tweets: %.0f" % (i,len(train_corpus)))
        vectors[i] = GM.buildDocumentVector(doc, num_of_dim, global_vectors)
    
    train_document_vecs = np.concatenate(vectors)
    train_document_vecs = sklearn.preprocessing.scale(train_document_vecs)

    labels = HL.create_labels(nr_pos_tweets, nr_pos_tweets)
 
    model_scores= run_k_fold(neural_nets_functions, train_document_vecs, labels, epochs, n_folds,patience)
    
    return model_scores


def model_checkpoint_callback(save_filename, verbose_,):
    
    return keras.callbacks.ModelCheckpoint(
        save_filename,
        monitor='val_loss',
        verbose=verbose_, 
        save_best_only=True,
        save_weights_only=False, 
        mode='auto'
    )


def early_stopping_callback(patience_, verbose_):
    
    return keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=patience_,
        verbose = verbose_
    )
    

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
    train_document_vecs = np.concatenate([GM.buildDocumentVector(doc, num_of_dim, global_vectors) for doc in train_corpus])
    train_document_vecs = sk.preprocessing.scale(train_document_vecs)
    
    labels = HL.create_labels(nr_pos_tweets, nr_pos_tweets, kaggle=False)
       
    train_document_vecs, labels = HL.shuffle_data(train_document_vecs,labels)
    train_x, val_x, train_y, val_y = HL.split_data(train_document_vecs, labels, split)
       
    test_document_vecs = np.concatenate([GM.buildDocumentVector(doc, num_of_dim, global_vectors) for doc in predict_corpus])
    test_document_vecs = sk.preprocessing.scale(test_document_vecs)
   
    model = neural_net(num_of_dim)
    
    # Defining callbacks to be used under fitting process
    early_stopping = early_stopping_callback(patience_=patience, verbose_=1)
    model_checkpoint = model_checkpoint_callback("neural_model_prediction.hdf5", verbose_=1)
    
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
    model = load_model('neural_model_prediction.hdf5')
 
    prediction = model.predict(test_document_vecs)
   
    prediction = [1 if i > 0.5 else -1 for i in prediction]
           
    # Creating prediction
    ids = list(range(1,10000+1))
    HL.create_csv_submission(ids, prediction,kaggle_name)
 
    return prediction





# KEEP - USED IN EXPERIMENTAL PHASE 
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