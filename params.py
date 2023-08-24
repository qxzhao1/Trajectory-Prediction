#Directory pointers
root_dir = "/home/Desktop/prediction/"
examples_dir = root_dir + "examples/"
raw_data_dir = root_dir + "raw_data"
processed_data_dir = root_dir + "processed_data/"
models_dir   = root_dir + "models/"
graphs_dir = root_dir + "graphs/"
animation_dir = root_dir + "animations/"

#For now, we have to change some parts of the code to accomodate for testing of new simulation (validation set), but in the future, we should have the model_test code run off of a separate directory that has a validation set, rather than running from same set of data we used to train.

#When testing with other validation set (In case you want to test with completely different data)
# root_dir = "/Users/rush/Documents/mason0820/test_files/"
# raw_data_dir = root_dir + "test_raw_data/"
# processed_data_dir = root_dir+"test_processed_data/"
# examples_dir = root_dir + "test_examples/"
# models_dir = "/Users/rush/Documents/mason0820/models/"
# graphs_dir = "/Users/rush/Documents/mason0820/graphs/"
# animation_dir = "/Users/rush/Documents/mason0820/animations/"

#Preprocessing Hyperparameters
speed_threshold = 0.001
lane_threshold = 15
speed_sign_threshold = 11
turn_threshold = 7
change_threshold = 1.7

#Model Parameters
seq_length=30
pred_length=20
step_size=2
validation_ratio = 0.2
firstLSTM_hiddenStates = 100
secondLSTM_hiddenStates = 80
l2_reg = 0.005
steps_to_ignore = 5
num_epochs = 1
save_model_name = "_100_80_STI5" #Set this to save the model with this extension after training

#For Visualization of graphs, you need to create a directory under graphs_dir same name as testing model
mode = "Testing" # Training: training the model. Testing: testing model. Main reason for this is because we have a validation-ratio split, and to visualize the data, we want to only visualize and test from the validation-set. Therefore, we need to introduce the "mode", so that the load_raw_data function knows which raw data to load into sequences.
storeGraph = False #True: Store graphs.  False: Don't store. If you are planning on saving the graphs, you must have a directory named the same as "testing_model" under your graph_dir.
drawGraph = False #True: Draw Graph dynamically.  False: Don't Draw dynamically
saveDynamicGraph = False #True: Saves the dynamic graph.   False: Don't save
graphOffset = 0 #Offset of dynamic graphs, so that you they will not be on top of each other. Currently not used because I have plotted graphs of diff size so its clear enough as is
skipOffset = 13 #For dynamic plots, plot every "skipOffset" value, otherwise too slow
drawWhichSeq = [12] #List of sequences (vehicles 0-len(vehicles)) you want to draw dynamically
testing_model = "d2d_speed_seq30_pred20_100_80_Reg005_STI5" #Which model do you want to test/which directory to save your graph snapshots in.
testingDelta = 20 #Which delta you want to graph/evaluate. Must have a value from 1 to pred_length
   
#To run the code:

#To run models,
# 	1) Change the save_model_name above to whatever you would like your model to be named as. By default, the type (a2a, a2d, d2d) will be appended to the front of the name you give.
# 	2) Run the code from model.py using the train_model method
# 	3) Your model will be saved under model_dir, with the name you gave it. 

# To Visualize or test the data,
# 	1) Change the testing_model to whatever model name you want to test.
# 	2) Toggle if you want to store the graphs, which will go to graphs_dir
# 		2.5) Note here that you have to create the matching "testing_model" directory under graphs_dir first
# 	3) You can also draw dynamic graphs as you create them. This will draw them with respect to time frame. Currently there is a small issue here, after you draw the graph, you can "x" out of the graph window, but to proceed, you will need to press ctrl+c once in the terminal.
# 			3.5) drawWhichSeq is a list of graphs that you want to see, for example you can see seqNum 2,4,5 by inputting [2,4,5] in this list. 
# 			3.5.1) testingDelta is the delta which you want to sample/visualize on your graphs or evaluate the performance of. 

boundary = {
1 : [5, [[-1.32, -4.82, 316.7, 144.5], [60.78, 551.18, 111.7, 108.1], [585.48, 581.58, 328.7, 179.2]]],
2 : [5, [[2.38, -1.32, 316.7, 144.5], [60.78, 551.18, 115.5, 111.7], [581.58, 577.88, 328.7, 179.2]]],
3 : [5, [[6.58, 2.38, 316.7, 144.5],[60.78, 551.18, 119.3, 115.5], [577.88, 573.88, 328.7, 179.2]]],
4 : [16, [[590.18, 586.48, 137.9, 326.1], [521.88, 23.38, 106.8, 103.1], [-9.92, -6.12, 192.6, 337.6]]],
5 : [16, [[593.88, 590.18, 137.9, 326.1], [521.88, 23.38, 103.1, 99.5], [-13.62, -9.92, 192.6, 337.6]]],
6 : [16, [[597.88, 593.88, 137.9, 326.1], [521.88, 23.38, 99.5, 95.5], [-17.42, -13.62, 192.6, 337.6]]],
7 : [22, [[180.28, 386.48, 354.60, 358.2]]],
8 : [22, [[180.28, 386.48, 358.2, 361.8]]],
9 : [22, [[180.28, 386.48, 361.8, 365.70]]],
10 : [18, [[159.58, 384.58, 353.20, 349.7]]],
11 : [18, [[159.58, 384.58, 349.7, 346]]],
12 : [18, [[159.58, 384.58, 346, 341.9]]],
13 : [10, [[176.58, 395.58, 502.7, 506.15]]],
14 : [10, [[176.58, 395.58, 506.15, 509.8]]],
15 : [10, [[176.58, 395.58, 509.8, 513.8]]],
16 : [0, [[375.78, 164.08, 497.35, 501]]],
17 : [0, [[375.78, 164.08, 493.6, 497.35]]],
18 : [0, [[375.78, 164.08, 490, 493.6]]],
19 : [7, [[171.68, 551.18, 699.7, 703.3], [590.18, 586.48, 517.6, 631]]],
20 : [7, [[171.68, 551.18, 703.3, 707], [593.88, 590.18, 517.6, 631]]],
21 : [7, [[171.68, 551.18, 707, 710.8], [597.88, 593.88, 517.6, 631]]],
22 : [14, [[585.48, 581.58, 663.8, 539.4], [509.48, 166.78, 694.7, 698.3]]],
23 : [14, [[581.58, 577.88, 663.8, 539.4], [509.48, 166.78, 691, 694.7]]],
24 : [14, [[577.88, 573.88, 663.8, 539.4], [509.48, 166.78, 687, 691]]],
25 : [4, [[597.88, 593.88, 366.9, 474.2]]],
26 : [4, [[593.88, 590.18, 366.9, 474.2]]],
27 : [4, [[590.18, 586.48, 366.9, 474.2]]],
28 : [8, [[585.48, 581.58, 475.40, 391.7]]],
29 : [8, [[581.58, 577.88, 475.40, 391.7]]],
30 : [8, [[577.88, 573.88, 475.40, 391.7]]],
31 : [20, [[417.78, 414.28, 468.2, 376.1]]],
32 : [20, [[421.38, 417.78, 468.2, 376.1]]],
33 : [20, [[425.38, 421.38, 468.2, 376.1]]],
34 : [9, [[412.78, 409.08, 387.80, 479.7]]],
35 : [9, [[409.08, 405.38, 387.80, 479.7]]],
36 : [9, [[405.38, 401.68, 387.80, 479.7]]],
37 : [3, [[451.28, 562.36, 354.60, 358.2]]],
38 : [3, [[451.28, 562.36, 358.2, 361.8]]],
39 : [3, [[451.28, 562.36, 361.8, 365.70]]],
40 : [21, [[430.38, 551.98, 353.20, 349.7]]],
41 : [21, [[430.38, 55198, 349.7, 346]]],
42 : [21, [[430.38, 551.98, 346, 341.9]]],
43 : [6, [[441.98, 561.48, 502.7, 506.15]]],
44 : [6, [[441.98, 561.48, 506.15, 509.8]]],
45 : [6, [[441.98, 561.48, 509.8, 513.8]]],
46 : [11, [[551.98, 437.88, 497.35, 501]]],
47 : [11, [[551.98, 437.88, 493.6, 497.35]]],
48 : [11, [[551.98, 437.88, 490, 493.6]]],
49 : [7, [[146.78, 143.08, 527.3, 665.3]]],
50 : [7, [[150.58, 146.78, 527.3, 665.3]]],
51 : [7, [[154.38, 150.58, 527.3, 665.3]]],
52 : [1, [[141.78, 138.18, 484.3, 387.4]]],
53 : [1, [[138.18, 134.48, 484.3, 387.4]]],
54 : [1, [[134.48, 130.28, 484.3, 387.4]]],
55 : [2, [[146.78, 143.08, 375.5, 464.2]]],
56 : [2, [[150.58, 146.78, 375.5, 464.2]]],
57 : [2, [[154.38, 150.58, 375.5, 464.2]]],
58 : [12, [[141.78, 138.18, 674.5, 530.4]]],
59 : [12, [[138.18, 134.48, 674.5, 530.4]]],
60 : [12, [[134.48, 130.28, 674.5, 530.4]]],
61 : [23, [[104.78, 21.78, 694.7, 698.3], [-9.92, -6.12, 383, 669.5]]],
62 : [23, [[104.78, 21.78, 691, 694.7], [-13.62, -9.92, 383, 669.5]]],
63 : [23, [[104.78, 21.78, 687, 691], [-17.42, -13.62, 383, 669.5]]],
64 : [13, [[6.58, 2.38, 617.3, 378.3], [45.78, 125.98, 699.7, 703.3]]],
65 : [13, [[2.38, -1.32, 617.3, 378.3], [45.78, 125.98, 703.3, 707]]],
66 : [13, [[-1.32, -4.82, 617.3, 378.3], [45.78, 125.98, 707, 710.8]]],
67 : [19, [[28.18002441, 115.08, 354.60, 358.2]]],
68 : [19, [[28.18002441, 115.08, 358.2, 361.8]]],
69 : [19, [[28.18002441, 115.08, 361.8, 365.70]]],
70 : [15, [[18.08, 114.18, 353.20, 349.7]]],
71 : [15, [[18.08, 114.18, 349.70, 346]]],
72 : [15, [[18.08, 114.18, 346, 341.9]]]
}


