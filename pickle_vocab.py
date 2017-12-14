#!/usr/bin/env python3
import pickle


def main():
    vocab = dict()
    with open('vocab_cut.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open('vocab11.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()


    
    
    
def pickle_vocab(vocab_cut_filename_txt, vocab_pkl_filename_pkl):
    vocab = dict()
    with open(vocab_cut_filename_txt) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(vocab_pkl_filename_pkl, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
