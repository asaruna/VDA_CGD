
import numpy as np
import os
from matplotlib import pyplot
from numpy import interp
import sklearn, tensorflow
import xlsxwriter 
import xlrd
from sklearn import svm 
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import gzip
import pandas as pd
import pdb
import random
from random import randint
import scipy.io
from tensorflow.keras import layers
from keras.layers import Input, Dense
from keras.engine.training import Model
from keras.models import Sequential, model_from_config,Model
from keras.layers.core import  Dropout, Activation, Flatten
from keras.layers import PReLU
from keras.utils import np_utils, generic_utils
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import LSTM
from keras.layers import Embedding
from keras import regularizers
from keras.constraints import maxnorm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deepforest import CascadeForestClassifier
from genetic_selection import GeneticSelectionCV
from sklearn.preprocessing import LabelEncoder
from keras.layers.serialization import activation
from keras.models import Sequential, model_from_config,Model
from sklearn.neighbors import KNeighborsClassifier



def prepare_data(seperate=False):
    print ("loading data")
    v_fea = np.loadtxt("virus.txt",dtype=float,delimiter="\t")
    assoc_dv = np.loadtxt("virusdrugassoc.txt",dtype=int,delimiter="\t")
    d_fea  = np.loadtxt("drug.txt",dtype=float,delimiter="\t")
    link_number = 0
    train = []  
    testfnl= []       
    label1 = []
    label2 = []
    label22=[]
    ttfnl=[]   
   
    for i in range(0, assoc_dv.shape[0]):  
        for j in range(0, assoc_dv.shape[1]):         
            if interaction[i, j] == 1:                      
                label1.append(assoc_dv[i,j])             
                link_number = link_number + 1                    
                d_fea_tmp = list(d_fea[i]) 
                v_fea_tmp = list(v_fea[j])
                tmp_fea = (d_fea_tmp,v_fea_tmp)   
                train.append(tmp_fea)                     
            elif assoc_dv[i,j] == 0:                     
                label2.append(assoc_dv[i,j])             
                dfea_tmp1 = list(d_fea[i])             
                v_fea_tmp1 = list(v_fea[j])
                test_fea= (d_fea_tmp1,v_fea_tmp1) 
                testfnl.append(test_fea)                    
    m = np.arange(len(label2))           
    np.random.shuffle(m)
    
    for x in m:
        ttfnl.append(testfnl[x])
        label22.append(label2[x])   
    
    for x in range(0, link_number):      
        tfnl= ttfnl[x]                                    
        lab= label22[x]                                      
        train.append(tfnl)                                  
        label1.append(lab)    
  
    return np.array(train), label1   
  
    
def transfer_array_format(data):    
    formated_matrix1 = []
    formated_matrix2 = []
    
    for val in data:     
        formated_matrix1.append(val[0])  
        formated_matrix2.append(val[1])  
    return np.array(formated_matrix1), np.array(formated_matrix2)

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
        y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
      return y, encoder

def features(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) 
    model = Sequential()
    
    return_sequences=True
    
    model.add(Convolution2D(16, (1,16),strides=(2,1), activation='relu', padding='same',                        data_format='channels_last',name='layer1_con1',input_shape=(1,253,1)))
    model.add(Dropout(0.5)) 
    
    model.add(MaxPooling2D(pool_size=(1,2),strides=(2,1), padding = 'same', data_format='channels_last',name = 'layer1_pool'))
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu',  name='Dense-2')) 
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', name='OutputLayer'))
    
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics = ['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=15, batch_size=10,validation_split=0.1)
    
    from keras.models import Model
    model2 = Model(inputs=model.input, outputs=model.get_layer('Dense-2').output)
    out = model2.predict(x) 
    print("extracted feature shape",out.shape)

def GA(feacnn,y_train):
  estimator = KNeighborsClassifier(3)
  G1model = GeneticSelectionCV(
    estimator, cv=5, verbose=2,
    scoring="accuracy", max_features=64,
    n_population=70, crossover_proba=0.8,
    mutation_proba=0.005, n_generations=200,
    crossover_independent_proba=0.5,
    mutation_independent_proba=0.04,
    tournament_size=3, n_gen_no_change=25,
    caching=True,
    n_jobs=-1)
    
  G1model = G1model.fit(feacnn,y_train)
  print('features',G1model.support_)
  
  encoder = LabelEncoder()
  fea_final=encoder.fit_transform(G1model.support_)
  count_arr = np.bincount(fea_final)    
  count_arr[1]
  n=[]
  Gfea=[]
  c=0
  for i in range(feacnn.shape[0]):
    n=[]
    feature=list(feacnn[i])
    for c in range(feacnn.shape[1]):
      if (fea_final[c]==1):
        n.append(feature[c])
    Gfea.append(np.array(n))
  G2model = Sequential()
  G2model.add(layers.Dense(64,activation='relu',input_shape=(Gfea.shape[1],)))
  G2model.add(layers.Dense(64,activation='relu'))
  G2model.compile(optimizer='adam',loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy']
  
  G2model.summary()
  G2model_out=G2model.predict(Gfea)
  return G2model,G2model_out,fea_final


def transfer_label_from_prob(proba):
    label = [1 if val>=0.6 else 0 for val in proba]
    return label


def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    
    if tp == 0 and fp == 0:
        precision = 0
        MCC = 0
        f1_score=0
        sensitivity =  float(tp)/ (tp+fn)
        specificity = float(tn)/(tn + fp)
    else:
        precision = float(tp)/(tp+ fp)
        sensitivity = float(tp)/ (tp+fn)
        specificity = float(tn)/(tn + fp)
        MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        f1_score= float(2*tp)/((2*tp)+fp+fn)

    return acc, precision, sensitivity, specificity, MCC,f1_score

if __name__=="__main__":
    dvacgd()
                  
def dvacgd():
    X, labels= prepare_data(seperate = True)     
    X_data1, X_data2 = transfer_array_format(X)    
                  
    X_data1= np.concatenate((X_data1, X_data2 ), axis = 1) 
                  
    y, encoder = preprocess_labels(labels)
    num = np.arange(len(y))   
    np.random.shuffle(num)
    X_data1 = X_data1[num]
    y = y[num]
    real__labels = []
    for val in y:
       if val[0] == 1:             
          real__labels.append(0)
       else:
          real__labels.append(1)    
    t=0
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    X_data1 = features(X_data1,real__labels)   
                  
    Gmodel,G2,G3=GA(X_data1,real__labels)
                  
    num_cross_val = 5
    all_performance_DF = []
    all_labels = []
    all_prob = {}

    all_prob[0] = []
    all_prob[1] = []
    all_prob[2] = []
    all_prob[3] = []
    all_averrage = []
    for fold in range(num_cross_val):
        train1 = np.array([x for i, x in enumerate(G2) if i % num_cross_val != fold])
        test1 = np.array([x for i, x in enumerate(G2) if i % num_cross_val == fold])
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
       
        
        real_labels = []               
        for val in test_label:
            if val[0] == 1:             
                real_labels.append(0)
            else:
                real_labels.append(1)

        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)
        class_index = 0
        class_index = class_index + 1
        
        prefilter_train = train1
        prefilter_test = test1
        m1 = CascadeForestClassifier(random_state=1,verbose=0,n_jobs=-1)
        estimators = [RandomForestClassifier(random_state=1,n_jobs=-1,n_estimators=100,max_depth=10),
                 RandomForestClassifier(random_state=1,n_jobs=-1,n_estimators=100,max_depth=15),
                 XGBClassifier(random_state=1,n_jobs=-1,n_estimators=100),
                 ExtraTreesClassifier(random_state=1,n_jobs=-1,n_estimators=100)]
        
        m1.set_estimator(estimators)
        m1.fit(prefilter_train, train_label_new
        
        proba = transfer_label_from_prob(ae_y_pred_prob) 
              
        acc, precision, sensitivity, specificity, MCC, f1_score = calculate_performace(len(real_labels), proba,  real_labels)
      
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)

        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        
        all_performance_DF.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score,f1_score])
        t =t+1  
        
        pyplot.plot(fpr,tpr,label= 'ROC fold %d (AUC = %0.4f)' % (t, auc_score))
        mean_tpr += interp(mean_fpr, fpr, tpr) 
        mean_tpr[0] = 0.0
                
        pyplot.xlabel('False positive rate')
        pyplot.ylabel('True positive rate')
        pyplot.title('Receiver Operating Characteristic curve: Five-Fold CV')
        pyplot.legend()
    
    mean_tpr /= num_cross_val
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print (np.mean(np.array(all_performance_DF), axis=0))
    Mean_Result=[]
    Mean_Result= np.mean(np.array(all_performance_DF), axis=0)
    print ('----' * 10)
    
    print('Mean-Accuracy=', Mean_Result[0],'\n Mean-precision=',Mean_Result[1])
    print('Mean-Sensitivity=', Mean_Result[2], '\n Mean-Specificity=',Mean_Result[3])
    print('Mean-MCC=', Mean_Result[4],'\n' 'Mean-auc_score=',Mean_Result[5])
    print('Mean-Aupr-score=', Mean_Result[6],'\n' 'Mean_F1=',Mean_Result[7])
    print ('----' * 10)
    
    pyplot.plot(mean_fpr, mean_tpr,'--' ,linewidth=3.5,label='Mean ROC (AUC = %0.4f)' % Mean_Result[5])
    pyplot.legend()  
    pyplot.show()
                  
