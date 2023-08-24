
import numpy as np
import codecs, json
import glob
import random
from params import *

# def print_example(inputs, outputs):
#     for i in range(len(inputs)):
#         print inputs[i], "--->", outputs[i]


# def print_examples(inputs, outputs):
#     for i in range(len(inputs)):
#         print "example ", i, ":"
#         print_example(inputs[i], outputs[i])


def load_sample_examples(dtype, n):
    files = glob.glob(examples_dir+dtype+"/" + "*.json")
    inputs, outputs = [], []
    for i in range(n):
        file = random.choice(files)
        with open(file) as infile:
            json_obj = json.load(infile)
        idx = random.choice(range(len(json_obj[0])))
        inputs.append(json_obj[0][idx])
        outputs.append(json_obj[1][idx])
    np_inputs  = np.asarray(inputs)
    np_outputs = np.asarray(outputs)
    return np_inputs, np_outputs

def print_sample_examples(dtype, n):
    inputs, outputs = load_sample_examples(dtype, n)
    print_examples(inputs, outputs)


# seq:    x0, x1, x2, x3, ...
# input:  x0, x1, x2, x3, ...
# out[0]: x1, x2, x3, x4, ...
# seq[1]: out[0][0]
def gen_example_a2a(seq, locations, start):
    inputs = seq[start:start+seq_length].tolist()
    outputs = []
    for k in range(seq_length):
        outputs.append(locations[start+1+k:start+1+k+pred_length].flatten().tolist())
    return inputs, outputs


# seq:    x0, x1, x2, x3, ...
# diff:   d1, d2, d3, d4, ...
# input:  x0, x1, x2, x3, ...
# out[0]: d1, d2, d3, d4, ...
# seq[1]: x0+out[0][0]
def gen_example_a2d(seq, locations, start):
    inputs = seq[start:start+seq_length].tolist()
    outputs = []
    for k in range(seq_length):
        outputs.append(np.diff(locations, axis=0)[start+k:start+k+pred_length].flatten().tolist())
    return inputs, outputs


# seq:    x0, x1, x2, x3, ...
# diff:   d1, d2, d3, d4, ...
# input:  d1, d2, d3, d4, ...
# out[0]: d2, d3, d4, d5, ...
# seq[2]: x1+out[0][0]
def gen_example_d2d(seq, locations, start):
    location_diff = np.diff(locations, axis=0) #[start:start+seq_length]
    for i in range(1,len(seq)):
        seq[i][1] = location_diff[i-1][0]
        seq[i][2] = location_diff[i-1][1]
    outputs = []
    inputs = seq[1:len(seq)]
    for k in range(seq_length):
        outputs.append(location_diff[start+1+k:start+1+k+pred_length].flatten().tolist())
    inputs = inputs[start:start+seq_length].tolist()
    return inputs, outputs

def gen_example_ad2d(seq, locations, start):
    location_diff = np.diff(locations, axis=0) #[start:start+seq_length]
    seq = seq.tolist()
    for i in range(1,len(seq)):
        seq[i].extend([seq[i][1], seq[i][2]])
        seq[i][1] = location_diff[i-1][0]
        seq[i][2] = location_diff[i-1][1]
    outputs = []
    inputs = seq[1:len(seq)]
    for k in range(seq_length):
        outputs.append(location_diff[start+1+k:start+1+k+pred_length].flatten().tolist())
    inputs = inputs[start:start+seq_length]
    return inputs, outputs

def gen_example(seq, locations, start, dtype):
    if dtype=="a2a":
        return gen_example_a2a(seq, locations, start)
    elif dtype=="a2d":
        return gen_example_a2d(seq, locations, start)
    elif dtype=="d2d":
        return gen_example_d2d(seq, locations, start)
    elif dtype=="ad2d":
        return gen_example_ad2d(seq, locations, start)
    else:
        assert(0)


def gen_examples_for_track_id(features, locations, dtype):
    start = 0
    inputs, outputs = [], []
    while start + seq_length + pred_length + 1 <= len(features):
        i, o = gen_example(features, locations, start, dtype)
        inputs.append(i)
        outputs.append(o)
        start = start + step_size

    return inputs, outputs


def gen_examples_from_file(file_path, dtype_):
    locations = np.loadtxt(file_path, dtype=float, usecols=[3,4], delimiter = ',')
    features_for_tid = np.loadtxt(file_path, dtype=float, usecols=[2,3,4,6,7,8,9,10,11],  delimiter = ',')
    inputs, outputs = [], []
    i, o = gen_examples_for_track_id(features_for_tid, locations, dtype_)
    inputs.extend(i)
    outputs.extend(o)
    return inputs, outputs


def gen_examples(dtype):
    files = glob.glob(processed_data_dir + "*.txt")
    for i, file in enumerate(files):
        inputs, outputs = gen_examples_from_file(file, dtype)
        print("i: ", i)
        print("input dimensions: ", len(inputs), "x", len(inputs[0]), "x", len(inputs[0][0]))
        print("output dimensions: ", len(outputs), "x", len(outputs[0]), "x", len(outputs[0][0]))
        with open(examples_dir+dtype+"_seq"+str(seq_length)+"_pred"+str(pred_length)+"/"+str(i)+'.json', 'w') as outfile:
            json.dump([inputs, outputs], outfile)


def check_size(dtype):
    print(dtype)
    files = glob.glob(examples_dir+dtype+"/"+"*.json")
    for file in files:
        with open(file) as infile: 
            example = json.load(infile)
        print("inputs: ", len(example[0]), "x", len(example[0][0]), "x", len(example[0][0][0]))
        print("outputs: ", len(example[1]), "x", len(example[1][0]), "x", len(example[1][0][0]))


# gen_examples("a2a")
# gen_examples("a2d")
gen_examples("d2d")
# gen_examples("ad2d")
check_size("d2d_speed_seq30_pred20")
