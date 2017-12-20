import pickle
import helpers as HL


## Dette er bare et utkast, bør helst være mye bedre.. 
#Loading the preprocessed corpus: 
final_corpus_ngram_stopwords=pickle.load( open( "stopword100_corpus_n2_SHM_E_SN_H_HK.pkl", "rb" ) )
unseen_test_set=final_corpus_ngram_stopwords[2500000:]

# Trenger hjelp til å lage test_document_vecs

#Loading the trained model: 
model = load_model('best_neural_model_save.hdf5')



pred=model.predict(test_document_vecs)

pred_ones=[]
    for i in pred:
        if i> 0.5:
            pred_ones.append(1)
        else:
            pred_ones.append(-1)
           
        
        
#CREATING SUBMISSION
ids = list(range(1,10000+1))
HL.create_csv_submission(ids, pred_ones,'powerpuffz_kagglescore.csv')



