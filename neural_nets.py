import random as rn
import tensorflow as tf
from keras import backend as K
import sklearn as sk
from keras.layers import *
from keras.layers.core import *
import keras

from numpy.random import seed

# ALL MODELS REQUIRE TO SET SEED() FOR REPRODUCABLE RESULTS

def basic_model(input_dimensions):
    
    seed(1337) # Needed for reproducable results 
    
    model = keras.models.Sequential()
    model.add(Dense(100, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def basic_model_adam(input_dimensions):
    
    seed(1337)

    model = keras.models.Sequential()
    model.add(Dense(100, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def wide_model(input_dimensions):
    
    seed(1337)

    model = keras.models.Sequential()
    model.add(Dense(150, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def deep_1_model(input_dimensions):
    
    seed(1337)

    model = keras.models.Sequential()
    model.add(Dense(100, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(60, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def deep_2_model(input_dimensions):
    
    seed(1337)

    model = keras.models.Sequential()
    model.add(Dense(100, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(60, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def deep_HB(input_dimensions):
    
    seed(1337)

    model = keras.models.Sequential()
    model.add(Dense(150, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(80, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(40, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(40, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    #model.add(Dense(1,kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def deep_HB_dropout(input_dimensions):
    
    seed(1337)

    model = keras.models.Sequential()
    model.add(Dense(150, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(80, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(40, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(40, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    #model.add(Dense(1,kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model