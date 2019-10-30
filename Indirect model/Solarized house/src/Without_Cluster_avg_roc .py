# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 23:19:38 2018

@author: mhrahman
"""

import keras
from keras.layers import Activation, LSTM,Input, Dense,Concatenate,Dropout,GRU
from data_loader import load_data, load_data_coded
from keras.models import Model, optimizers
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
import sys
import numpy as np
import pandas as pd
import math
from sklearn.metrics import confusion_matrix, roc_curve,auc, accuracy_score, precision_score, f1_score, recall_score
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
#from mlxtend.evaluate import confusion_matrix
import uuid
import os

MODEL = 'LSTM'

Static_data = 2
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
lstm_out= LSTM(units= 128,activation='tanh')(main_input)
#lstm_out = LSTM(units=32,activation='tanh')(lstm_out)
lstm_out=Dropout(0.2)(lstm_out)
#lstm_out=Dense(128,activation='relu')(lstm_out)
#lstm_out=Dense(128,activation='relu')(lstm_out)
lstm_out=Dense(action_count,activation='relu')(lstm_out)
main_output=Activation('softmax')(lstm_out)
model=Model(inputs=[main_input],outputs=main_output)
print(model.summary())

#model_define
lr=[0.1]
epochs= 30

sgd=optimizers.SGD(lr=lr,momentum=0.9,nesterov=True)
adam=optimizers.Adam(lr=lr,beta_1=0.9,beta_2=0.999,epsilon=1e-08)
rmsprop=optimizers.rmsprop(lr=lr,rho=0.9,epsilon=None)
model.compile(optimizer=sgd,loss='categorical_crossentropy')

model_weights_file = '.tmp_nn_weights.{}'.format(uuid.uuid4())
model.save_weights(model_weights_file)

#Training _Testing
seq_count=len(actions)
LOOCV=seq_count
Fold= 5
fold_size=int(math.ceil(float(seq_count)/Fold))
print('Training on {} sample in {} fold'.format(seq_count,Fold))

train_accs=[]
test_accs=[]
pred_acc=[]
roc_info = []
metric_ = []

for i in range(Fold):
    print('\nFold {}'.format(i+1))
    model.load_weights(model_weights_file)
    
    start = i * fold_size
    end = start + fold_size if i + 1 < Fold else seq_count
    
    x_train = [x for j, x in enumerate(actions) if j < start or j >= end]
    x_test = actions[start:end]

    max_length = np.amax([len(x) for x in x_train])
    indices = np.arange(1, max_length)

    def calc_acc(series,skip=0):
        loss = 0
        count = 0

        max_len = np.amax([len(x) for x in series])
        for k in range(1, max_len):
            feat = np.array([x[0:k] for x in series if len(x) > k])
            lab = np.array([x[k] for x in series if len(x) > k])
            pred = np.argmax(model.predict([feat]), axis=1)
            actual = np.argmax(lab, axis=1)
            count += len(pred)
            loss += sum([x != y for x, y in zip(pred, actual)])
            
        if count == 0:
            return 1.0
        else:
            return 1. - loss / float(count)
        
# ROC curve function
            
    def roc_calc(series,skip=0):
        pred_f=[]
        lab_f=[]
        max_len = np.amax([len(x) for x in series])
        for k in range(1,max_len):
            feat = np.array([x[0:k] for x in series if len(x) > k])
            lab = np.array([x[k] for x in series if len(x) > k])
            pred = (model.predict([feat]))
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
    
    def metric(series, skip = 0):
        f1_scores= []
        precision_scores = []
        recall_scores = []
        max_len = np.amax([len(x) for x in series])
        for k in range(1, max_len):
            feat = np.array([x[0:k] for x in series if len(x) > k])
            lab = np.array([x[k] for x in series if len(x) > k])
            pred = np.argmax(model.predict([feat]), axis=1)
            actual = np.argmax(lab, axis=1)
            f1 = f1_score(actual,pred,average='weighted')
            f1_scores.append(f1)
            pr = precision_score(actual,pred,average= 'weighted')
            precision_scores.append(pr)
            recall = recall_score(actual,pred, average= 'weighted')
            recall_scores.append(recall)
        return (np.average(f1_scores),np.average(precision_scores),np.average(recall_scores))
            
    
#Prediction each iteration--------------------------------- 
        
    def pred_each(series,skip=0):
        loss = 0
        count = 0
        x=0
        ac=[]
        max_len = np.amax([len(x) for x in series])
        for k in range(1, max_len):
            feat = np.array([x[0:k] for x in series if len(x) > k])
            lab = np.array([x[k] for x in series if len(x) > k])
            pred = np.argmax(model.predict([feat]), axis=1)
            actual = np.argmax(lab, axis=1)
            count = len(pred)
            loss = sum([x != y for x, y in zip(pred, actual)])
            x=1-loss/float(count)
            ac.append(x)
        return ac

    train_acc = calc_acc(x_train)
    test_acc = calc_acc(x_test)
    print('train_acc={}, test_acc={}'.format(train_acc, test_acc))

    got_nan = False

    tr_acc=[]
    te_acc=[]
    for epoch in range(epochs):
        np.random.shuffle(indices)
        for index, k in enumerate(indices):
            feat = np.array([x[0:k] for x in x_train if len(x) > k])
            lab = np.array([x[k] for x in x_train if len(x) > k])
            h = model.fit(x=[feat], y=lab, verbose=False)

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

        train_acc = calc_acc(x_train)
        test_acc = calc_acc(x_test)
#        predi=pred_each(x_test)
        tr_acc.append(train_acc)
        te_acc.append(test_acc)
        print('\ntrain_acc={}, test_acc={}'.format(train_acc, test_acc))

    train_accs.append(train_acc)
    test_accs.append(test_acc)
    lstm_roc=roc_calc(x_test)
    roc_info.append(lstm_roc)
#    pred_acc.append(predi)
    met = metric(x_test)
    metric_.append(met)


roc_socre = []
for i in roc_info:
    roc_socre.append(i[2]["macro"])
    a=np.mean(roc_socre)
    s = np.std(roc_socre)

        
print('\nAverage: train_acc={}, test_acc={}, AUC = {}, A_std= {}'.format(np.average(train_accs), np.average(test_accs),a,s))
#print('\nTrain data: Max={},Min={},Std={}'.format(np.max(train_accs),np.min(train_accs),np.std(train_accs)))    
print('\nTest data: Max={},Min={},Std={}'.format(np.max(test_accs),np.min(test_accs),np.std(test_accs)))
# creating data frame using FPR and TPR
path = r'C:\Users\mhrahman\Desktop\Neural network\LSTM_static\Solarized house\Output'
#Use OS library (chdir) to set location
os.chdir(path)
Accuracy = [['Train',np.average(train_accs)],['Test',np.average(test_accs)], ['Test_std', np.std(test_accs)],['AUC', a], ['AUC_std', s]]
Table = pd.DataFrame(Accuracy)
Table.to_csv('Table {}.csv'.format(MODEL),index = False)

Design_process = pd.DataFrame([roc_info[0][2],roc_info[1][2],roc_info[2][2],roc_info[3][2],roc_info[4][2]])
Design_process_avg = pd.DataFrame([dict(Design_process.mean())]).T
Design_process_std = pd.DataFrame([dict(Design_process.std())]).T
Design_process_avg.to_csv('Design_process.csv',index = True)
Design_process_std.to_csv('Design_process_std.csv',index = True)

for i in range(len(roc_info)):
    df = pd.DataFrame()
    df['FPR'] = roc_info[i][0]["macro"]
    df['TPR'] = roc_info[i][1]["macro"]
    df.to_csv("Fold {}.csv".format(i+1),index = False)
me_df = pd.DataFrame(metric_, columns=['F1', 'Precision', 'Recall'])
me_df.to_csv('Score.csv', index = False)

for i in range(len(roc_info)):
    plt.rcParams["font.family"]="Times New Roman"
    plt.plot(roc_info[i][0]["macro"], roc_info[i][1]["macro"],
             label='Average ROC curve (area = {0:0.2f})'
                   ''.format(roc_info[i][2]["macro"]),
             color='deeppink', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','black','brown','burlywood'])
    design_process=['Formulation','Synthesis','Evaluation','Reformulation 2','Analysis',
                    'Reformulation 3','Reformulation 1']
    for j,color in zip(range(action_count),colors):
        plt.plot(roc_info[i][0][j], roc_info[i][1][j], color=color,
        label='ROC curve of class {0} (area = {1:0.2f})'
        ''.format(design_process[j], roc_info[i][2][j]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for LSTM model',fontsize=12)
    plt.legend(loc="lower right",fontsize=11)
    plt.savefig('LSTM_ROC_{}'.format(i),dpi=1000) 
    plt.show()
    plt.close('all')
    
