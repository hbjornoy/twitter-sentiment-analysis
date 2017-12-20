# Eksternal libraries
import csv
import pickle
import time
import keras as K

# internal imports
import helpers as HL

# Loading pre-processed document vectors for test-set. 
test_document_vecs =  pickle.load(open("final_document_vectors.pkl", "rb" ))

#Loading neural net model
model = K.models.load_model('final_model_for_kaggle.hdf5')

#Predicting on test set with neural net model
prediction = model.predict(test_document_vecs)

#Convert results to kaggle format ( -1, 1 )
prediction = [1 if i > 0.5 else -1 for i in prediction]
        
#CREATING SUBMISSION
ids = list(range(1,10000+1))
HL.create_csv_submission(ids, prediction,'powerpuffz_kagglescore.csv')

input("\n","Predictions made: powerpuffz_kagglescore.csv", "\n", "Press any key to exit...")