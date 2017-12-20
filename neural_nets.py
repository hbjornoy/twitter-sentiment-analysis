import random as rn
import tensorflow as tf
from keras import backend as K
import sklearn as sk
from keras.layers import *
from keras.layers.core import *
import keras

from numpy.random import seed

# ALL MODELS REQUIRE TO SET SEED() FOR REPRODUCABLE RESULTS


def basic_model_adam(input_dimensions):
    
    seed(1337)

    model = keras.models.Sequential()
    model.add(Dense(100, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def complex_model(input_dimensions):
    """
    This is our final neural net model that provided the best result. It is made through trial and error experimentation.
    """
    
    # for reprodusable models
    seed(1337)
    
    model = keras.models.Sequential()
    model.add(Dense(500, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(150, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
    
    
def dynamic_dense(input_dimensions, width, depth, dropout_rate=0.1, activation='relu', funnel=False):
    """
    This function is made to quikly generate a wide variaty of dense hidden layers.
    It is used in the trial and error fase of figuring out the best parameters for our final model which is named XXXX
    not compatible with current crossvalidation
    """
    # for reprodusable models
    seed(1337)
    
    model = keras.models.Sequential()
    # iterates through and adds layers to the model
    for layer_nr in range(0, depth):
        model.add(Dense(width, input_dim=input_dimensions, kernel_initializer='normal', activation=activation))
        model.add(Dropout(dropout_rate))
        if funnel is not False:
            width = int(np.ceil(funnel*width))
    
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
       
    
    