import os
import random 
import numpy as np
import itertools
from numpy import loadtxt
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout, Input, MaxPool2D
from keras.layers import Conv1D, MaxPool1D
from keras import regularizers
from keras.layers.merge import concatenate
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_scor
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import precision_recall_curve
########################### Loading train, test, and validation independent data test to train and test multimodal deep learning model
cell='neuron_ex_10%'
label=np.load('label_{}.npy'.format(cell))
train1=np.load('train_index_{}_1.npy'.format(cell))
train2=np.load('train_index_{}_2.npy'.format(cell))
train3=np.load('train_index_{}_3.npy'.format(cell))
train4=np.load('train_index_{}_4.npy'.format(cell))
train5=np.load('train_index_{}_5.npy'.format(cell))
indep_test1=np.load('indep_test_index_{}_1.npy'.format(cell))
indep_test2=np.load('indep_test_index_{}_2.npy'.format(cell))
indep_test3=np.load('indep_test_index_{}_3.npy'.format(cell))
indep_test4=np.load('indep_test_index_{}_4.npy'.format(cell))
indep_test5=np.load('indep_test_index_{}_5.npy'.format(cell))
valid_indep1=random.sample(list(indep_test1),500)
valid_indep2=random.sample(list(indep_test2),500)
valid_indep3=random.sample(list(indep_test3),500)
valid_indep4=random.sample(list(indep_test4),500)
valid_indep5=random.sample(list(indep_test5),500)
#################################
SEED=12345
test_scores=[]
valid_score=[]
train_score=[]
mm=10
for ii in range(0,5): ######## the indeice of five fold cross validation
    valid=valid_tests[ii]
    train=trains[ii]
    indep_test=indep_tests[ii]
    vectors=np.load('proximity_{}.npy'.format(bb))
    train_vectors=vectors[train].reshape(len(train),5,1,1)
    train_matrix=mat1[train].reshape(len(train),mm,mm,1)
    y_train_vectors=to_categorical(label[train])
    indep_test_vectors=vectors[indep_test].reshape(len(indep_test),5,1,1)
    indep_test_matrix=mat1[indep_test].reshape(len(indep_test),mm,mm,1)
    y_indep_test_vectors=to_categorical(label[indep_test])
    valid_vectors=vectors[valid].reshape(len(valid),5,1,1)
    valid_matrix=mat1[valid].reshape(len(valid),mm,mm,1)
    y_valid_vectors=to_categorical(label[valid])

    os.environ['PYTHONHASHSEED']=str(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)

    input_vectors = Input(shape=(5,1,1))
    input_matrix = Input(shape=(mm,mm,1))
    # the first channel operates on the first input (proximity features)
    x1 =  Dense(5, activation="relu")(input_vectors)
    x2=   Flatten()(x1)
    x3=   Dense(4, activation="relu")(x2)
    x4=   Dense(3, activation="relu")(x3)
    x =   Dense(2, activation="relu")(x4)
    
    # the second channel opreates on the second input (co-expression features)
    y1 = Conv2D(filters=36,kernel_size=(4,4),input_shape=(mm,mm,1),padding='same')(input_matrix)
    y2 = MaxPool2D(pool_size=(2,2),strides=2)(y1)
    y3 = Conv2D(filters=64,kernel_size=(4,4),padding='same')(y2)
    y4 = MaxPool2D(pool_size=(2,2),strides=2)(y3)
    y5=  Dropout(rate=0.1)(y4)
    y6=  Flatten()(y5)
    y =  Dense(50)(y6)
    
    # combine the output of the two branches
    combined = concatenate([x, y])
    
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z1 = Dense(50, activation="relu")(combined)
    z2=  Dropout(rate=0.1)(z1)
    z3 = Dense(20, activation="relu")(z2)
    z4 = Dense(10, activation="relu")(z3)
    z =  Dense(2, activation="sigmoid")(z4)
    
    # our model will accept the inputs of the two branches and
    # then output a single value
    modeli= Model(inputs=[input_vectors,input_matrix], outputs=z)
    modeli.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    batch_size=200
    modeli.fit([train_vectors,train_matrix],y_train_vectors,validation_data=([valid_vectors,valid_matrix],y_valid_vectors),batch_size=batch_size,epochs=12)
    batch_size=len(valid_vectors)
    
    batch_size=len(indep_test_matrix)
    scores=modeli.predict([indep_test_vectors,indep_test_matrix],batch_size=batch_size,verbose=1)
    
    fpr1, tpr1, threshold1=metrics.roc_curve(label[indep_test],scores[:,1],pos_label=1) # evaluation
    roc=auc(fpr1,tpr1)
    precision, recall, thresholds1_LR=precision_recall_curve(label[indep_test],scores[:,1],pos_label=1)
    pr=auc(recall,precision)
    print('roc-dep_test',roc) 
    print('pr-_dep_test',pr)





