import csv
import numpy as np
import glove_module as GV

def create_corpus(inputfiles):
    """ Given files, it returns the corpus.
    Input: 
        iputfiles: A list of filenames: ["filename1.txt", "filename2.txt"]. If only one filename, it must be given as
        ["filename1.txt"] and not "filename1.txt".
    Output:
        corpus: A corpus containing all files, in order: filename1, filename2, etc
        file_lenghts: A list of the lenght of each file. 
    """
    file_lengths=[]
    corpus=[]
    for file in inputfiles:
        with open(file,'rb') as infile:
            for line in infile:
                corpus.append(line[:-1])
        sum_so_far=sum(file_lengths)
        new_lenght=len(corpus)
        file_lengths.append(new_lenght-sum_so_far)
                    
    return corpus, file_lengths


def create_labels(nr_pos_lines,nr_neg_lines,kaggle=False):
    """
    Input: 
        nr_pos_lines: Number tweets from positive training data
        nr_neg_lines: Number tweets from negative training data
    Output: 
        labels: A array containing labels where 1 corresponds to positive training data and 
        0 corresponds to negative training data. 
    """
    nr_lines_total=nr_pos_lines+nr_neg_lines
    labels = np.zeros(nr_lines_total);
    labels[0:nr_pos_lines]=1;
    if kaggle:
        labels[nr_pos_lines:nr_lines_total]=-1; 
    else:
        labels[nr_pos_lines:nr_lines_total]=0;
    return labels


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


def create_csv_submission(ids, y_pred, name):
    """ Creates csv file
    TAKEN FROM ML COURSE: https://github.com/epfml/ML_course/blob/master/projects/project1/scripts/proj1_helpers.py
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

            
def get_corpus(full=False, test=False, inputfiles=None):
    
    """ Creating corpus and associated constants.
    Input: 
        full: if true, the retunred corpus contains all 2 510 000 tweets
        test: if true, the returned corpus contains the 210 000 test- tweets. 
        inputfiles: if any given, and test= False and full=Fase, the corpus will be created based on the given files. 
        Must be given as: inputfiles=['filename.txt', 'filename2.txt'] or ['file.txt']. 
    
    Outputs: 
        full_corpus: A corpus made of the desired files, 
        nr_pos_tweets: If test of full or inputfiles with 3 files are given, nr_pos_tweets is the number of tweets in
        the first file (which for test and full is the positive labeled tweets). Otherwise it is 0. 
        nr_neg_tweets: If test of full or inputfiles with 3 files are given, nr_neg_tweets is the number of tweets in 
        the second file (which for test and full is the negative labeled tweets). Otherwise it is 0. 
        total_training_tweets: The sum of the two above. 
    
    """
    
    training_set_pos = "train_pos.txt" 
    training_set_neg = "train_neg.txt"
    training_set_pos_full = "train_pos_full.txt"
    training_set_neg_full = "train_neg_full.txt"
    test_set = "test_data.txt"
    
    if test:
        inputfiles=[training_set_pos,training_set_neg,test_set]

    elif full:
        inputfiles=[training_set_pos_full,training_set_neg_full,test_set]
    
    if len(inputfiles)==3:

        full_corpus, file_lengths=create_corpus(inputfiles)
        nr_pos_tweets = file_lengths[0]
        nr_neg_tweets = file_lengths[1]
        total_training_tweets =file_lengths [0]+file_lengths[1]
    else: 
        nr_pos_tweets=0
        nr_neg_tweets=0
        total_training_tweets=0
        
    return full_corpus, nr_pos_tweets, nr_neg_tweets, total_training_tweets


def get_global_vectors(dimension): 
    """ Returns pre-trained Glove-Embeddings, given dimension. 
    Input: 
        dimension: The desired dimension in the returned global vectors. Can be 50,100 or 200. 
    
    Output: 
        global_vectors: The pre-trained global vectors,with desired dimension, on the gensim object form. 
    """
    if dimension==50:
        global_vectors=GV.create_glove_model("gensim_global_vectors_50dim.txt")
    elif dimension ==100: 
        global_vectors=GV.create_glove_model("gensim_global_vectors_100dim.txt")
    elif dimension== 200: 
        global_vectors=GV.create_glove_model("gensim_global_vectors_200dim.txt")
    elif dimension==25: 
        global_vectors=GV.create_glove_model("gensim_global_vectors_25dim.txt")
    else: 
        print('not valid dimension')
        global_vectors=-1
    return global_vectors     

   
            
        
            