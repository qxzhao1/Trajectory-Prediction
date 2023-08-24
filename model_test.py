import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from copy import deepcopy
from keras import regularizers
import json
import glob
import random
import model as Model
from params import *
import math
import visualization as Visual


def load_raw_sample():
    files = glob.glob(processed_data_dir + "*.txt")
    file_path = random.choice(files)
    sequences = np.loadtxt(file_path, dtype=float, usecols=[2,3,4,6,7,8,9,10,11], delimiter = ',')
    return sequences

#Since each file only contains information for 1 car, we can just append the features to sequences for all files. 
def load_processed_data():
    files = glob.glob(processed_data_dir + "*.txt")
    sequences = []
    for i, file in enumerate(files): 
        if mode == "Testing" and i < (1 - validation_ratio) * len(files):
            continue
        features = np.loadtxt(file, dtype=float, usecols=[2,3,4,6,7,8,9,10,11],  delimiter = ',')
        sequences.append(features)
    return sequences

# seq:    x0, x1, x2, x3, ...
# input:  x0, x1, x2, x3, ...
# out[0]: x1, x2, x3, x4, ...
# seq[1]: out[0][0]
def run_model_a2a(model, seq, verbose=0, seqNum=0):
    model.reset_states()
    seq_, pred_seq = [], []
    for i in range (seq.shape[0]-testingDelta):
        if storeGraph == False:
            seq_.append([seq[testingDelta+i][1]-graphOffset, seq[testingDelta+i][2]-graphOffset])
        else:
            seq_.append([seq[testingDelta+i][1], seq[testingDelta+i][2]])
        x = seq[i].reshape(1, 1, 9)
        pred = model.predict(x)[0][0]
        pred_x = pred[2*testingDelta-2]
        pred_y = pred[2*testingDelta-1]

        pred_seq.append([pred_x, pred_y])

    return seq_, pred_seq


# seq:    x0, x1, x2, x3, ...
# diff:   d1, d2, d3, d4, ...
# input:  x0, x1, x2, x3, ...
# out[0]: d1, d2, d3, d4, ...
# seq[1]: x0+out[0][0]
def run_model_a2d(model, seq, verbose=0, seqNum=0):
    model.reset_states()
    seq_, pred_seq = [], []
    for i in range (seq.shape[0]-testingDelta):
        if storeGraph == False:
            seq_.append([seq[testingDelta+i][1]-graphOffset, seq[testingDelta+i][2]-graphOffset])
        else:
            seq_.append([seq[testingDelta+i][1], seq[testingDelta+i][2]])
        x = seq[i].reshape(1, 1, 9)
        pred = model.predict(x)[0][0]
        pred_x, pred_y = seq[i][1], seq[i][2] 
        if verbose>0:
            print (seq[i], "--->", pred)
        for j in range(testingDelta):
            pred_x += pred[2*j]
            pred_y += pred[2*j+1]

        pred_seq.append([pred_x, pred_y])

    return seq_, pred_seq


def get_d2d_seq_diff(seq):
    seq_diff = deepcopy(seq)
    for i in range(len(seq_diff)-1, 0, -1):
        seq_diff[i][1] -= seq_diff[i-1][1]
        seq_diff[i][2] -= seq_diff[i-1][2]
        # seq_diff[i].extend([seq[i][1], seq[i][2]]) #This was for testing ad2d
    return seq_diff[1:]


def calcRelativeErrorPerFrame(pred_x, pred_y, seq, i):
    Ax = abs(seq[i+1+testingDelta][1] - seq[i+1][1])
    Ay = abs(seq[i+1+testingDelta][2] - seq[i+1][2])
    Ex = abs(abs(pred_x - seq[i+1][1]) - Ax)
    Ey = abs(abs(pred_y - seq[i+1][2]) - Ay)
    Exy = math.sqrt(Ex**2 + Ey**2)
    Axy = math.sqrt(Ax**2 + Ay**2)
    return Exy, Axy, Ex, Ey, Ax, Ay

# seq:    x0, x1, x2, x3, ...
# diff:   d1, d2, d3, d4, ...
# input:  d1, d2, d3, d4, ...
# out[0]: d2, d3, d4, d5, ...
# seq[2]: x1+out[0][0]
def run_model_d2d(model, seq, verbose, seqNum=0):
    model.reset_states()
    seq_diff = get_d2d_seq_diff(seq)
    seq_ , pred_seq = [], []
    allFrameError, allFrameTrue, allXError, allYError, allXActual, allYActual = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    numberOfFramesProcessed = len(seq_diff)-testingDelta-1
    for i in range(numberOfFramesProcessed): 
        seq_.append([seq[testingDelta+1+i][1], seq[testingDelta+1+i][2]])
        # if storeGraph == False: #This section was meant to give an offest to the graph so that overlapping graphs are not confusing.
        #     seq_.append([seq[testingDelta+1+i][1]-graphOffset, seq[testingDelta+1+i][2]-graphOffset])
        # else: 
        #     seq_.append([seq[testingDelta+1+i][1], seq[testingDelta+1+i][2]])
        x = seq_diff[i].reshape(1, 1, 9)
        pred = model.predict(x)[0][0]
        if verbose>0:
            print (x[0][0], [pred[2*testingDelta-2], pred[2*testingDelta-1]])

        pred_x, pred_y = seq[i+1][1], seq[i+1][2]

        for j in range(testingDelta):
            pred_x += pred[2*j]
            pred_y += pred[2*j+1]

        tempError, tempActual, tempXError, tempYError, tempXActual, tempYActual = calcRelativeErrorPerFrame(pred_x, pred_y, seq, i)

        allFrameError += tempError
        # allFrameTrue += tempActual  #this was for metric where allFrameError/allFrameTrue
        allXError += tempXError
        allYError += tempYError
        # allXActual += tempXActual   #For testing other metric. 
        # allYActual += tempYActual

        pred_seq.append([pred_x, pred_y])

    # return seq_, pred_seq, allFrameError/allFrameTrue, allXError/allXActual, allYError/allYActual
    return seq_, pred_seq, allFrameError/numberOfFramesProcessed, allXError/numberOfFramesProcessed, allYError/numberOfFramesProcessed

def test_model(model, run_model, verbose):
    sequences = load_processed_data()
    allRelativeError, allREX, allREY = 0.0, 0.0, 0.0
    processedSeqCount = 0
    for seqNum, seq in enumerate(sequences):
        if (drawGraph == True) and (seqNum not in drawWhichSeq):
            continue
        true, pred, perCarRelativeError, perCarREX, perCarREY = run_model(model, seq, verbose, seqNum)
        allRelativeError += perCarRelativeError
        # allREX += perCarREX
        # allREY += perCarREY
        print("**********************************")
        print("seqNum: ",seqNum, "seq: ", len(seq),"x", len(seq[0]), "  True: ", len(true), "x",len(true[0]), "  Pred: ", len(pred), "x",len(pred[0]))
        print("Relative Error in this car was : ", perCarRelativeError)
        print("RelativeX Error in this car was : ", perCarREX)
        print("RelativeY Error in this car was : ", perCarREY)
        print("**********************************")

        if storeGraph == True:
            Visual.graph_values(true, pred, seqNum)

        if drawGraph == True:
            Visual.graph_dynamic_values(true, pred, seqNum)

        processedSeqCount = seqNum+1

    allRelativeError /= processedSeqCount
    return allRelativeError
    # allREX /= processedSeqCount
    # allREY /= processedSeqCount
    # return allRelativeError, allREX, allREY


#Used mainly for comparing multiple models with the same hyperparameters, just different input sources. 
def printResults(model, model1, model2, totalDeltas):
    total_a2d, total_d2d, total_a2a = 0.0, 0.0, 0.0

    for delta in range(1,totalDeltas+1):    
        result_a2d = test_model(model, run_model_a2d, delta, 0)
        result_d2d = test_model(model1, run_model_d2d, delta, 0)
        result_a2a = test_model(model2, run_model_a2a, delta, 0)

        total_a2d += result_a2d
        total_d2d += result_d2d
        total_a2a += result_a2a

        print("\n\n\nTarget Diff being evaluated: ", delta)
        print("model_a2d: ", result_a2d)
        print("model_d2d: ", result_d2d)
        print("model_a2a: ", result_a2a)

    print("\n\n\n------------------------------")
    print("Total Average for a2d: ", total_a2d/totalDeltas)
    print("Total Average for d2d: ", total_d2d/totalDeltas)
    print("Total Average for a2a: ", total_a2a/totalDeltas)
    print("------------------------------\n\n\n")

verbose = 0
model = Model.load_model(testing_model)
print(test_model(model, run_model_d2d, verbose))
print("model: ", testing_model, "  delta: ", testingDelta)

# model = Model.load_model("a2d_30_30")
# model1 = Model.load_model("d2d_30_30")
# model2 = Model.load_model("a2a_30_30")
# print("_30_30")
# printResults(model, model1, model2, 5)

# model = Model.load_model("d2d_100_60_noReg")
# print("d2d_100_60_noReg: ", test_model(model, run_model_d2d, 1, 0))
# model = Model.load_model("d2d_30_30_noReg")
# print("d2d_30_30_noReg: ", test_model(model, run_model_d2d, 1, 0))
# model = Model.load_model("a2d_100_60_noReg_STI5")
# print("a2d_100_60_noReg_STI5: ", test_model(model, run_model_a2d, 1, 0))


