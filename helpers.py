import csv
import numpy as np
import glove_module as GV
import pickle

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

def creating_n_grams_corpus(n_gram, corpus):
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
def add_n_grams(line, n_gram):
    
    """
    Input: 
    line: A tweet
    n_gram: a number, the number of words that are to be grouped together
    
    Output: 
    b' '.join(n_grams): A line, containing the words from the input line grouped together in groups consisting of n_grams words. 
    Each group is separated by a space, the words within a group is separated by a hyphen. 
    
    For a line like 'this is an example', the output would be:
    n_gram=2 '_-this this-is is-an an-example example-_'
    n_gram=3 '_-_-this _-this-is this-is-an is-an-example an-example-_ example-_-_'
    
    """

    n_grams = []
    words = line.split()
    if n_gram is not 1:
        #need to treat the first n_gram-1 words diffrently:
        for first_words in range(n_gram-1):
            temp=[]
            number_of_blanks= n_gram-(first_words+1)
            for blank in range(number_of_blanks):
                temp.append(b'|_')
            words_to_fill_in=first_words+1
            for w in range(words_to_fill_in):
                temp.append(words[w])
                if w is not (words_to_fill_in-1):
                                temp.append(b'_')
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
                    temp.append(b'_')
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
                temp.append(b'_')
                        
            for blank in range(number_of_blanks):
                temp.append(b'|')
                if blank is not number_of_blanks-1:
                    temp.append(b'_')
            
            n_grams.append(b''.join(temp))

    return b' '.join(n_grams)

def creating_n_grams_corpus(n_gram, corpus):
    """
    Input: 
    n_gram: A number, HJELP desired number of maximum number of words grouped together?
    corpus: A corpus
    
    
    Output: 
    new_corpus: A corpus, containing all word groups consisting of up to n_gram words. 
    """
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

            
def get_corpus(full=False, test=False, inputfiles=None):
    
    """
    Input: 
    full: if true, the retunred corpus contains all 2 510 000 tweets
    test: if true, the returned corpus contains the 210 000 test- tweets. 
    inputfiles: if any given, and test= False and full=Fase, the corpus will be created based on the given files. 
    Must be given as: inputfiles=['filename.txt', 'filename2.txt'] or ['file.txt']. 
    
    Outputs: 
    full_corpus: A corpus made of the desired files, 
    nr_pos_tweets: If test of full or inputfiles with 3 files are given, nr_pos_tweets is the number of tweets in the first file
    (which for test and full is the positive labeled tweets). Otherwise it is 0. 
    nr_neg_tweets: If test of full or inputfiles with 3 files are given, nr_neg_tweets is the number of tweets in the second file
    (which for test and full is the negative labeled tweets). Otherwise it is 0. 
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
    """
    Input: 
    dimension: The desired dimension in the returned global vectors. can be 50,100 or 200. 
    
    Output: 
    global_vectors: The pre-trained global vectors,with desired dimension, on the gensim object form. 
    """
    if dimension==50:
        global_vectors=GV.make_glove("gensim_global_vectors_50dim.txt")
    elif dimension ==100: 
        global_vectors=GV.make_glove("gensim_global_vectors_100dim.txt")
    elif dimension== 200: 
        global_vectors=GV.make_glove("gensim_global_vectors_200dim.txt")
    elif dimension==25: 
        global_vectors=GV.make_glove("gensim_global_vectors_25dim.txt")
    else: 
        print('not valid dimension')
        global_vectors=-1
    return global_vectors   

class MacOSFile(object):
    """
    This object makes us the objects able to get pickled in Mac event though they superseed the limit of 4GB.
    """

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))

   
            
        
            