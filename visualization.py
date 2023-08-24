from keras.layers import LSTM
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed
import numpy as np
from keras.models import load_model
import json
import glob
import keras.backend as K
from keras.models import model_from_json
from keras.utils import plot_model
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from keras.models import Sequential
from matplotlib import pyplot as plt
from matplotlib import animation as animation
import model as Model
from params import *
import time 
import random

#Separates the actual[[X1,Y1],[X2,Y2]...] to actualX[X1,X2,X3..] and actualY[Y1,Y2,Y3...]
def separate_values(actual, predicted):
    actualX, actualY, predX, predY = [],[],[],[]
    for i in range(len(actual)):
        actualX.append(actual[i][0])
        actualY.append(actual[i][1])
        predX.append(predicted[i][0])
        predY.append(predicted[i][1])

    return predX, predY, actualX, actualY

#Graphs the values and saves them to the graph_dir/testing_model directory. 
#The way the file number is named is by 'seq'+seqNum+'delta'+delta
#seqNum is same as vehicle number, and delta is which step in the future you are predicting (delta range is 1-10) 
def graph_values(actual, predicted, seqNum):
    predX, predY, actualX, actualY = [], [], [], []
    predX, predY, actualX, actualY = separate_values(actual, predicted)
    
    plt.figure(seqNum)
    plt.plot(predX, predY, "ro", actualX, actualY, "k.")
    plt.legend(["Predicted", "Actual"])
    axes = plt.gca()
    axes.set_ylim(50,750)
    axes.set_xlim(-50, 625)
    plt.title(testing_model+'/seq'+str(seqNum)+'delta'+str(testingDelta))
    plt.savefig(graphs_dir+testing_model+'/seq'+str(seqNum)+'delta'+str(testingDelta)+'.png')
    plt.close(seqNum)


#Creates the pred and actual arrays used for dynamic graphing. Allows the usage of skipOffset, which makes plotting faster, rather than plotting all the points.
def createFinal(predX, predY, actualX, actualY):
    pred, actual = [[],[]], [[],[]]
    for i in range(0, len(predX), skipOffset):
        pred[0].append(predX[i])
        pred[1].append(predY[i])
        actual[0].append(actualX[i])
        actual[1].append(actualY[i])

    return np.array(pred), np.array(actual)


#Part of graph_dynamic_values, updates the values to the graph.
def update_line(num, data, line, data2, line2):
    line.set_data(data[0, :num], data[1, :num])
    line2.set_data(data2[0, :num], data2[1, :num])
    return line, line2


#Graphs the values dynamically on the screen. You will see it in real time, and although it is slow, it seems to be the fastest I can print as of now. My next step is to see if I can skip some frames and print every 10th frame or so. 
#There is a small annoyance, where while running, in order to see the next graph you inputted in the drawWhichSeq list, you need to 'x' out the of graph, and also press ctrl+c ONCE in the terminal. This will allow the program to move on and plot out the next sequence in drawWhichSeq. 
def graph_dynamic_values(actual, predicted, seqNum):
    predX, predY, actualX, actualY = [], [], [], []
    predX, predY, actualX, actualY = separate_values(actual, predicted)
    data, data2 = createFinal(predX, predY, actualX, actualY)

    fig = plt.figure()
    predPlot = fig.add_subplot(111)
    l, = predPlot.plot([], [], "ro", label="Predicted")

    actPlot = predPlot.twinx()
    k, = actPlot.plot([], [], "k.", label="Actual")

    predPlot.legend([l,k], [l.get_label(), k.get_label()], loc=0)
    predPlot.set_xlim(-100,650)
    predPlot.set_ylim(0,750)
    actPlot.set_xlim(-100,650)
    actPlot.set_ylim(0, 750)
    plt.title(testing_model+'/seq'+str(seqNum)+'delta'+str(testingDelta))

    line_ani = animation.FuncAnimation(fig, update_line, frames=int(len(actual)/skipOffset), fargs=(data, l, data2, k), interval=100, blit=True, repeat=True, repeat_delay=None)
    if saveDynamicGraph == True:
        line_ani.save(animation_dir+testing_model+'_seq'+str(seqNum)+'delta'+str(testingDelta)+".mp4")
    plt.show()




