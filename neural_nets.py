import sklearn as sk
import keras
from keras.layers import *
from keras.layers.core import *
import tensorflow as tf
import random as rn
from keras import backend as K

def basic_model(input_dimensions):
    
    model = keras.models.Sequential()
    model.add(Dense(100, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def basic_model_adam(input_dimensions):
    model = keras.models.Sequential()
    model.add(Dense(100, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def wide_model(input_dimensions):
    model = keras.models.Sequential()
    model.add(Dense(150, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def deep_1_model(input_dimensions):
    model = keras.models.Sequential()
    model.add(Dense(100, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(60, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def deep_2_model(input_dimensions):
    model = keras.models.Sequential()
    model.add(Dense(100, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(60, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def deep_HB(input_dimensions):
    rn.seed(7)
    tf.set_random_seed(7)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
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


def recurrent_model(input_dimensions):
    # Start neural network
    model = keras.models.Sequential()

    # Add an embedding layer
    model.add(Embedding(input_dim=input_dimensions, output_dim=128))

    # Add a long short-term memory layer with 128 units
    model.add(LSTM(units=128))

    # Add fully connected layer with a sigmoid activation function
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile neural network
    model.compile(loss='binary_crossentropy',  # Cross-entropy
                  optimizer='Adam',  # Adam optimization
                  metrics=['accuracy'])  # Accuracy performance metric

    return model


def convolutional_model(input_dimensions):
    model = keras.models.Sequential()
    model.add(Conv1D(32, activation='elu', padding='same', input_shape=(133332, input_dimensions)))
    model.add(Conv1D(32, activation='elu', padding='same'))
    model.add(Conv1D(32, activation='elu', padding='same'))
    model.add(Conv1D(32, activation='elu', padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv1D(32, activation='elu', padding='same'))
    model.add(Conv1D(32, activation='elu', padding='same'))
    model.add(Conv1D(32, activation='elu', padding='same'))
    model.add(Conv1D(32, activation='elu', padding='same'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

    return model