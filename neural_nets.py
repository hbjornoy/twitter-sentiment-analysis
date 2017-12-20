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
    """ Returns a basic/simple neural network model, used as a baseline. 
    Input: 
        input_dimensions: The dimension of the global vectors. 
    
    Output: 
        model: A Keras sequencial neural net model. (referred to as basic/ simple neural net in report). 
    """
    
    """
    """
    
    seed(1337)

    model = keras.models.Sequential()
    model.add(Dense(100, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def complex_model(input_dimensions):
    """ Returns the final/complex neural network model, used for final results.  
        It is designed by trial and error experimentation.
    Input: 
        input_dimensions: The dimension of the global vectors. 
    
    Output: 
        model: A Keras sequencial neural net model. (referred to as final/complex neural net in report). 
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
    It is used in the trial and error fase of figuring out the best parameters for our final model (complex_model)
    Output model is not compatible with current crossvalidation
    
    Inputs: 
        input_dimensions: The dimension of the global vectors.
        width: Number of nodes in hidden layers.
        depth: Number og hidden layers.
        dropout_rate: Desired dropout rate. Default 0.1. 
        activation function: Activation function for hidden layers, default 'relu'. 
        funnel: Desired funnel rate. Default False. 
        
    Output: 
        model: A Keras sequencial neural net model. 
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
    
    