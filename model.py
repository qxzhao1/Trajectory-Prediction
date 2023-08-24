import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras import regularizers
import json
import glob
import random
from params import *

def load_data(file_path):
    inputs, outputs = [], []
    files = glob.glob(file_path + "*.json")
    for file in files:
        with open(file) as infile:
            json_obj = json.load(infile)
        inputs  = inputs  + json_obj[0]
        outputs = outputs + json_obj[1]
    np_inputs  = np.asarray(inputs)
    np_outputs = np.asarray(outputs)
    return np_inputs, np_outputs


def mean_error(y_true, y_pred):
    shp = tf.shape(y_true)
    ones = K.ones(shape=(1, shp[1]-steps_to_ignore, shp[2]))
    zeros = K.zeros(shape=(1, steps_to_ignore, shp[2]))
    mask = tf.concat([zeros, ones], 1)
    zeros2 = K.zeros(shape=(2*pred_length-2))
    ones2 = K.ones(shape=(2))
    mask2 = tf.concat([zeros2, ones2], 0)
    y_true_masked = K.abs(y_true)*mask*mask2
    error_masked = K.abs(y_pred-y_true)*mask*mask2
    return K.sum(error_masked)/K.sum(y_true_masked)


def loss_func(y_true, y_pred):
    shp = tf.shape(y_true)
    ones = K.ones(shape=(1, shp[1]-steps_to_ignore))
    zeros = K.zeros(shape=(1, steps_to_ignore))
    mask = tf.concat([zeros, ones], 1)
    temp = K.mean(K.abs(y_pred - y_true), axis=2) * mask
    temp = K.mean(temp, axis=-1)
    return temp


def train_model(dtype):
    dataInputs, expectedOutputs = load_data(examples_dir+dtype+'/')
    model = Sequential()
    model.add(LSTM(firstLSTM_hiddenStates, input_shape=(dataInputs[0].shape), kernel_regularizer=regularizers.l2(l2_reg), return_sequences=True))
    model.add(LSTM(secondLSTM_hiddenStates, kernel_regularizer=regularizers.l2(l2_reg), return_sequences=True))
    model.add(Dense(2*pred_length, kernel_regularizer=regularizers.l2(l2_reg), activation=None))

    model.compile(optimizer="Adam", loss=loss_func, metrics=[mean_error])
    history = model.fit(dataInputs, expectedOutputs, epochs=num_epochs, batch_size=512, validation_split=validation_ratio, verbose=2)

    model2 = Sequential()
    model2.add(LSTM(firstLSTM_hiddenStates, batch_input_shape=(1,1,9), return_state=False, return_sequences=True, stateful=True))
    model2.add(LSTM(secondLSTM_hiddenStates, return_state=False, return_sequences=True, stateful=True))
    model2.add(Dense(2*pred_length, activation=None))
    model2.set_weights(model.get_weights())
    model2.save(models_dir+dtype+save_model_name)
    return model2

def load_model(whichModel): 
    return keras.models.load_model(models_dir+whichModel)


# model = train_model("a2a")
# model = train_model("a2d")
# model = train_model("d2d")
# model = train_model("a2a_seq20_pred20")
# model = train_model("a2d_seq20_pred20")
model = train_model("d2d_seq30_pred20")
