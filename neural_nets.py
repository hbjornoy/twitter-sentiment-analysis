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


def wide_model(input_dimensions):
    
    seed(1337)

    model = keras.models.Sequential()
    model.add(Dense(300, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
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
    model.add(Dropout(0.1))
    model.add(Dense(100, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(80, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(40, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(20, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(40, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    #model.add(Dense(1,kernel_initializer='random_uniform', bias_initializer='zeros'))
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
        
    """
    OTHER IDEAS:
    - decreasing, uniform width
    - decreasing dropoutrate?
    - last activation
    - optimizer
    - activation
    """
    
    


"""
def conv1d(input_dimensions):
    
    seed(1337)
    model = keras.models.Sequential()
    
    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters=128,
                     input_shape=(input_dimensions,1),
                     kernel_size=7,
                     padding='valid',
                     activation='relu'))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())
    # We add a vanilla hidden layer:
    model.add(Dense(100))
    # model.add(Dropout(0.2))
    model.add(Activation('relu'))
    
    model.add(Dense(40, kernel_initializer='normal', activation='relu'))
    
    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
    

def embedded_model(input_dimensions):

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(labels_index), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    
    
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(labels_index), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    # happy learning!
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=2, batch_size=128)
    """
    
    
    
    
    