# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 23:19:38 2018

@author: mhrahman
"""

import keras
from keras.layers import Activation, LSTM,Input, Dense,Concatenate,Dropout,GRU
from data_loader import load_data, load_data_coded
from keras.models import Model
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
import sys
import numpy as np
import pandas as pd
import math
from keras import optimizers
import tensorflow as tf
from sklearn.metrics import multilabel_confusion_matrix,roc_curve,auc
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
import uuid
from sklearn.metrics import f1_score


cpu_core = 1
conf = tf.ConfigProto(intra_op_parallelism_threads = cpu_core, inter_op_parallelism_threads = cpu_core)
session = tf.Session(config = conf)
K.set_session(session)

Static_data = 4
action_count,actions,cluster= load_data_coded(Static_data)

#padding 0 for uneven sequence lengths
Transposed=[]
for i in range(len(actions)):
    Transposed.append(actions[i].transpose())
padded=[pad_sequences(i,maxlen=620,padding='post') for i in Transposed]
f_actions=[]
for i in range(len(actions)):
    f_actions.append(padded[i].transpose())


#Model Building
main_input= Input(shape=(None,action_count),name='main_input')
Static_input=Input(shape=(6,),name='Static_input')
lstm_out= LSTM(units=256,activation='tanh')(main_input)
Static_out=Dense(units=128,activation='sigmoid')(Static_input)
merge=Concatenate(axis=-1)([lstm_out,Static_out])
merge=Dropout(0.2)(merge)
#merge=Dense(units=128)(merge)
merge=Dense(action_count)(merge)
main_output=Activation('softmax')(merge)
model=Model(inputs=[main_input,Static_input],outputs=main_output)
print(model.summary())


#model_define
lr=[.001]
epochs = 30

sgd=optimizers.SGD(lr=lr,momentum=0.9,nesterov=True)
adam=optimizers.Adam(lr=lr,beta_1=0.9, beta_2=0.999, epsilon=1e-8)
rmsprop=optimizers.rmsprop(lr=lr,rho=0.9,epsilon=None)
model.compile(optimizer=adam,loss='categorical_crossentropy')

model_weights_file = '.tmp_nn_weights.{}'.format(uuid.uuid4())
model.save_weights(model_weights_file)

#Training _Testing
seq_count=len(actions)
LOOCV=seq_count
Fold=4
fold_size=int(math.ceil(float(seq_count)/Fold))

train_accs=[]
test_accs=[]
pred_acc=[]
roc_info = []
for i in range(Fold):
    print('\nFold {}'.format(i+1))
    model.load_weights(model_weights_file)
    
    start = i * fold_size
    end = start + fold_size if i + 1 < Fold else seq_count

    x_train = [x for j, x in enumerate(actions) if j < start or j >= end]
    y_train = [x for j, x in enumerate(cluster) if j < start or j >= end]
    x_test = actions[start:end]
    y_test = cluster[start:end]

    max_length = np.amax([len(x) for x in x_train])
    indices = np.arange(1, max_length)

    def calc_acc(series_1,series_2,skip=0):
        loss = 0
        count = 0

        max_len = np.amax([len(x) for x in series_1])
        for k in range(1, max_len):
            ini_feat_1=[]
            ini_feat_2=[]
            for i,x in enumerate(series_1):
                if len(x)>k:
                    ini_feat_1.append(x[0:k])
                    ini_feat_2.append(series_2[i])
            feat_1=np.array(ini_feat_1)
            feat_2=np.array(ini_feat_2)
            #feat_1 = np.array([x[0:k] for x in series_1 if len(x) > k])
            #feat_2 = np.array(series_2[0:len(feat_1)])
            lab = np.array([x[k] for x in series_1 if len(x) > k])
            pred = np.argmax(model.predict([feat_1,feat_2]), axis=1)
            actual = np.argmax(lab, axis=1)
            count += len(pred)
            loss += sum([x != y for x, y in zip(pred, actual)])

        if count == 0:
            return 1.0
        else:
            return 1. - loss / float(count)
        
# ROC curve function
            
    def roc_calc(series_1, series_2,skip=0):
        pred_f=[]
        lab_f=[]
        max_len = np.amax([len(x) for x in series_1])
        for k in range(1, max_len):
            ini_feat_1=[]
            ini_feat_2=[]
            for i,x in enumerate(series_1):
                if len(x)>k:
                    ini_feat_1.append(x[0:k])
                    ini_feat_2.append(series_2[i])
            feat_1=np.array(ini_feat_1)
            feat_2=np.array(ini_feat_2)
            #feat_1 = np.array([x[0:k] for x in series_1 if len(x) > k])
            #feat_2 = np.array(series_2[0:len(feat_1)])
            lab = np.array([x[k] for x in series_1 if len(x) > k])
            pred = (model.predict([feat_1,feat_2]))
            pred_f.append(pred)
            lab_f.append(lab)
        pre=np.concatenate(pred_f)
        lab_=np.concatenate(lab_f)
        fpr=dict()
        tpr=dict()
        roc_auc=dict()
        for i in range(action_count):
            fpr[i],tpr[i],_=roc_curve(lab_[:,i],pre[:,i])
            roc_auc[i]=auc(fpr[i],tpr[i]) 
#        all_fpr=np.unique(np.concatenate([fpr[i] for i in range(action_count)]))
        mean_fpr = np.linspace(0,1,200)
        mean_tpr=np.zeros_like(mean_fpr)
        for i in range(action_count):
            mean_tpr +=interp(mean_fpr,fpr[i],tpr[i])
        mean_tpr/=action_count
        fpr["macro"]=mean_fpr
        tpr["macro"]=mean_tpr
        roc_auc["macro"]=auc(fpr["macro"],tpr["macro"])
        return (fpr,tpr,roc_auc)

#Prediction each iteration--------------------------------- 
        
    def pred_each(series_1,series_2,skip=0):
        loss = 0
        count = 0
        x=0
        ac=[]
        
        max_len = np.amax([len(x) for x in series_1])
        for k in range(1, max_len):
            ini_feat_1=[]
            ini_feat_2=[]
            for i,x in enumerate(series_1):
                if len(x)>k:
                    ini_feat_1.append(x[0:k])
                    ini_feat_2.append(series_2[i])
            feat_1=np.array(ini_feat_1)
            feat_2=np.array(ini_feat_2)
            #feat_1 = np.array([x[0:k] for x in series_1 if len(x) > k])
            #feat_2 = np.array(series_2[0:len(feat_1)])
            lab = np.array([x[k] for x in series_1 if len(x) > k])
            pred = np.argmax(model.predict([feat_1,feat_2]), axis=1)
            actual = np.argmax(lab, axis=1)
            count = len(pred)
            loss = sum([x != y for x, y in zip(pred, actual)])
            x=1-loss/float(count)
            ac.append(x)
        return ac        

    train_acc = calc_acc(x_train,y_train)
    test_acc = calc_acc(x_test,y_test)
    print('train_acc={}, test_acc={}'.format(train_acc, test_acc))

    got_nan = False

    tr_acc=[]
    te_acc=[]
    for epoch in range(epochs):
        np.random.shuffle(indices)
        for index, k in enumerate(indices):
            ini_feat_1=[]
            ini_feat_2=[]
            for i,x in enumerate(x_train):
                if len(x)>k:
                    ini_feat_1.append(x[0:k])
                    ini_feat_2.append(y_train[i])
            feat_1=np.array(ini_feat_1)
            feat_2=np.array(ini_feat_2)
            #feat_1 = np.array([x[0:k] for x in x_train if len(x) > k])
            #feat_2 = np.array(y_train[0:len(feat_1)])
            lab = np.array([x[k] for x in x_train if len(x) > k])
            #lab=np.delete(lab,np.s_[7:],axis=1)
            h = model.fit(x=[feat_1,feat_2], y=lab, verbose=False)

            if math.isnan(h.history['loss'][0]):
                print('\nWARNING: NaN occurred! Treating as an accuracy of 0.')
                got_nan = True
                break

            print('\repoch {} / {}, batch {} / {}'.format(epoch+1, epochs, index+1, len(indices)), end='')
            sys.stdout.flush()

        if got_nan:
            got_nan = False
            train_acc = 0
            test_acc = 0
            break

        train_acc = calc_acc(x_train,y_train)
        test_acc = calc_acc(x_test,y_test)
#        predi=pred_each(x_test,y_test)
        tr_acc.append(train_acc)
        te_acc.append(test_acc)
        print('\ntrain_acc={}, test_acc={}'.format(train_acc, test_acc))

    train_accs.append(train_acc)
    test_accs.append(test_acc)
    lstm_roc=roc_calc(x_test,y_test)
    roc_info.append(lstm_roc)
#   pred_acc.append(predi)
roc_socre = []
for i in roc_info:
    roc_socre.append(i[2]["macro"])
    a=np.mean(roc_socre)
print('\nAverage: train_acc={}, test_acc={}, AUC = {}'.format(np.average(train_accs), np.average(test_accs),a))
#print('\nTrain data: Max={},Min={},Std={}'.format(np.max(train_accs),np.min(train_accs),np.std(train_accs)))    
print('\nTest data: Max={},Min={},Std={}'.format(np.max(test_accs),np.min(test_accs),np.std(test_accs)))
# creating data frame using FPR and TPR
path = r'C:\Users\mhrahman\Desktop\Neural network\LSTM_static\Solarized house\Output'
#Use OS library (chdir) to set location
for i in range(len(roc_info)):
    df = pd.DataFrame()
    df['FPR'] = roc_info[i][0]["macro"]
    df['TPR'] = roc_info[i][1]["macro"]
    df.to_csv("Fold {}.csv".format(i+1),index = False)