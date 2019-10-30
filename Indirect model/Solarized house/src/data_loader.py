import json
import keras
import numpy as np
import os
import random

from keras import backend as K

def load_data_coded(aug=0, seq_dir='../data/csv/', shuffle=True):
    print("Dir=" + seq_dir)
    files = os.listdir(seq_dir)
    if shuffle:
        random.shuffle(files)

    sequences = []
    for filename in files:
        with open(seq_dir + os.sep + filename) as f:
            sequences.append(_load_seq_coded(f.readlines()))

    sequences, cluster,actions = _process_sequences_coded(sequences, aug)
    return len(actions),sequences,cluster

def _load_seq_coded(lines):
    return [x[0:-1] for x in lines[1:] if x[1:-2] != '']

def _load_clusters(lines):
    # extract just the data

    t1=[]
    for x in lines[1:]:
        if len(x)==29:
            r=x[5:-1]
        else:
            r = x[4:-1]
        t1.append(r)
    
    # divide on the commas
    t2 = [l.split(',') for l in t1]
    
    # convert to floating point numbers
    t3 = []
    for z in t2:
        r = [float(el) for el in z]
        t3.append(r)
    
    # convert to a one-hot representation
    t4=[keras.utils.to_categorical(i,7) for i in t3]
    #t4 = []
    #for z in t3:
    #    r = [keras.utils.to_categorical(y, 7) for y in z]
    #    t4.append(r) 
    # t4 is now a 38 (students) by 12 (cluster algorithms) by 7 (cluster ids) tensor
    #print("clusters: ")
    #print(t4)
    return t4

def _process_sequences_coded(sequences, aug):
    # read the clusters file
    clusters = []
    with open('../data/Cluster_house.csv') as f:
        clusters = _load_clusters(f.readlines())

    print("sequences length: ")
    print(len(sequences))
    actions = list(set([x for y in sequences for x in y]))
    sequences = [[actions.index(y) for y in x] for x in sequences]
    sequences = [keras.utils.to_categorical(x, len(actions)) for x in sequences]

    if aug == 0:
        return sequences, actions

    print("Augmenting the data with cluster method " + str(aug))

    # fuse the cluser data with the action data
    clust_alg = aug - 1;
    seqs = []
    for i in range(len(clusters)):
        clust = clusters[i]
        cc = clust[clust_alg]
        cc = cc[1:]
            #for 4 clusters
            #cc = cc[:-2]
            # for 5 clusters
            #cc = cc[:-1]
            #concat = np.concatenate((action,cc),axis=0)
            #concat=concat[:-1]
            
        seqs.append(cc)
        #seqs=[np.asmatrix(np.array(x)) for x in seqs]
        seqs=[np.array(x) for x in seqs]
        # try to remove the last column of zeros;

    # print the data to make sure we did it correctly
    #for i in range(38):
        #print("Before...\n")
        #for x in sequences[i]:
            #print(x)
        #print("After...\n")
        #for x in seqs[i]:
            #print(x)

    return sequences, seqs,actions

def load_data(act_file='../data/actions.txt', seq_dir='../data/json/', shuffle=True):
    with open(act_file) as f:
        actions = [x.strip() for x in f.readlines()]

    files = os.listdir(seq_dir)
    if shuffle:
        random.shuffle(files)

    sequences = []
    for filename in files:
        with open(seq_dir + os.sep + filename) as f:
            sequences.append(_load_seq(json.load(f), actions))

    action_seqs, reward_seqs = _process_sequences(sequences, actions)
    return len(actions), action_seqs, reward_seqs

def _load_seq(data, actions):
    action_set = set(actions)
    action_seq = []
    reward_seq = []

    reward = 0.
    for i, x in enumerate(data['Activities']):
        act = set(x.keys()) & action_set
        if len(act) > 0:
            action_seq.append(next(iter(act)))
            if 'EnergyAnnualAnalysis' in x and 'Net' in x['EnergyAnnualAnalysis']:
                reward = float(x['EnergyAnnualAnalysis']['Net']['Total']) - reward
                reward_seq.append(reward)
            else:
                reward_seq.append(0)

    return action_seq, reward_seq

def _process_sequences(sequences, actions):
    action_seqs = [[actions.index(y) + 1 for y in x[0]] for x in sequences]
    action_seqs = [keras.utils.to_categorical(x, len(actions) + 1) for x in action_seqs]

    low = np.amin([np.amin(x[1]) for x in sequences])
    high = np.amax([np.amax(x[1]) for x in sequences])
    reward_seqs = [(x[1] - low) / (high - low) + 1 for x in sequences]
    reward_seqs = [np.array([x * y for x, y in zip(l, m)]) for l, m in zip(action_seqs, reward_seqs)]

    return action_seqs, reward_seqs
