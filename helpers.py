import csv
import numpy as np

def create_corpus(inputfiles):
    """
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
                corpus.append(line)
        sum_so_far=sum(file_lengths)
        new_lenght=len(corpus)
        file_lengths.append(new_lenght-sum_so_far)
                    
    return corpus, file_lengths

def create_labels(nr_pos_lines,nr_neg_lines):
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
    labels[nr_pos_lines:nr_lines_total]=0;
    return labels


def add_n_grams(line, n):

    n_grams = []
    words = line.split()
    n_grams.append(b''.join([b'_',b'-',words[0]]))
    for i in range(len(words)-1):        
        n_grams.append(b''.join([words[i], b'-', words[i+1]]))
     
    n_grams.append(b''.join([words[-1], b'-', b"_"]))
    return b' '.join(n_grams)

def creating_n_grams_cropus(n_gram, corpus):
    new_corpus=[]
    for line in corpus:
        if line:
            line = add_n_grams(line, n_gram)
        new_corpus.append(line)
    
    return new_corpus



def create_csv_submission(ids, y_pred, name):
    """
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
            
            