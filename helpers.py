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
"""
def add_n_grams(line, n_gram):

    n_grams = []
    temp=[]
    words = line.split()
    #n_grams.append(b''.join([b'_',b'-',words[0]]))
    for i in range(len(words)-(n_gram)): 
        temp=[]
        index=i
        if n_gram is not 1:
            for j in range(n_gram):
                temp.append(words[index])
                if j is not (n_gram-1):
                    temp.append(b'-')
                index+=1
            n_grams.append(b''.join(temp))
        else: 
            n_grams.append(b''.join([words[i]]))
    #n_grams.append(b''.join([words[-1], b'-', b"_"]))
    #n_grams.append(b''.join([words[-1]]))
    return b' '.join(n_grams)

def creating_n_grams_cropus(n_gram, corpus):
    new_corpus=[]
    
    for line in corpus:
        lines=[]
        if line:
            for i in range(n_gram):
                new_line = add_n_grams(line, i+1)
                lines.append(new_line)
        new_corpus.append(b' '.join(lines))
    
    return new_corpus

"""
def ngram(text,grams):
    model=[]
    count=0
    for token in text[:len(text)-grams+1]:
       model.append(text[count:count+grams])
       count=count+1
    return model

def add_n_grams(line, n_gram):

    n_grams = []
    words = line.split()
    if n_gram is not 1:
        #need to treat the first n_gram-1 words diffrently:
        for first_words in range(n_gram-1):
            temp=[]
            number_of_blanks= n_gram-(first_words+1)
            for blank in range(number_of_blanks):
                temp.append(b'_-')
            words_to_fill_in=first_words+1
            for w in range(words_to_fill_in):
                temp.append(words[w])
                if w is not (words_to_fill_in-1):
                                temp.append(b'-')
            n_grams.append(b''.join(temp))
    
    
    for i in range(len(words)-(n_gram)+1): 
        #creating empty temporary list
        temp=[]
        #initializing index:
        index=i
        if n_gram is not 1:    
            #want to combine n_gram words in temp, with a '-' between each word.
            for j in range(n_gram):
                temp.append(words[index])
                if j is not (n_gram-1):
                    temp.append(b'-')
                index+=1
            #want to combine everything as a bit string.
            n_grams.append(b''.join(temp))
        else: 
            n_grams.append(b''.join([words[i]]))
            
    if n_gram is not 1:
        #need to treat the final n_gram-1 words diffrently:
        for last_words in range(n_gram-1):
            temp=[]
            number_of_blanks= n_gram-(last_words+1)
            words_to_fill_in=last_words+1
            
   
            for w in range(words_to_fill_in,0,-1):
                temp.append(words[-(w)])
                temp.append(b'-')
                        
            for blank in range(number_of_blanks):
                temp.append(b'_')
                if blank is not number_of_blanks-1:
                    temp.append(b'-')
            
            n_grams.append(b''.join(temp))

    return b' '.join(n_grams)

def creating_n_grams_cropus(n_gram, corpus):
    new_corpus=[]
    
    for line in corpus:
        lines=[]
        if line:
            for i in range(n_gram):
                if len(line.split())>n_gram:
                    new_line = add_n_grams(line, i+1)
                    lines.append(new_line)
        new_corpus.append(b' '.join(lines))
    
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
            
            