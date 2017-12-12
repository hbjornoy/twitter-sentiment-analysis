from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
import time
import sklearn as sk
import numpy as np
import keras
import neural_nets as NN

def create_gensim_word2vec_file(path_to_glove_folder):
    """
    :param path_to_glove_folder: Go to Stanfords website https://nlp.stanford.edu/projects/glove/ and download their twitterdataset,
            put it in the same folder as this function and write the path to it as the input of this function
    :return: nothing but creates a .txt-file with the gensim object, afterwards you can delete the original glove-files
            and keep the created gensim___.txt files and use the function load_gensim_global_vectors(path_to_global_vectors) to load them in.
    """

    for filename in os.listdir(path_to_glove_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(path_to_glove_folder, filename)
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
    """
    TRY NORMALIZING gloVe.syn0norm BEFORE INPUTTING

    :param neural_nets_functions:
    :param global_vectors:
    :param processed_corpus:
    :param total_training_tweets:
    :param nr_pos_tweets:
    :return:
    """
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




