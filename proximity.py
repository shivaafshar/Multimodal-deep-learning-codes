import numpy as np
from itertools import combinations
import itertools
import scipy.stats as stats
import csv
import ast
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold
import networkx as nx
from functools import reduce
import random
from numpy import * 

def proximity(name_list,graphs):
         b=[]
               
         b_neighbor=[]    
         b_neighbor.append(len(sorted(nx.common_neighbors(G,name_list[iiii][0],name_list[iiii][1]))))
         
         jc=[]
         f=nx.jaccard_coefficient(G, ebunch=[name_list_t[iiii]])
         for u, v, p in f:
            jc.append(p)    
          
         pa=[]
         f=nx.preferential_attachment(G, ebunch=[name_list_t[iiii]])
         for u, v, p in f:
            pa.append(p)     
          
         # print(i)
         aa=[]
         f=nx.adamic_adar_index(G, ebunch=[name_list_t[iiii]])
         for u, v, p in f:
           aa.append(p)
         
              #print(i)
         short=[]     
         if nx.has_path(G, source=name_list[iiii][0], target=name_list[iiii][1]):
                    path = nx.shortest_path_length(G, source=name_list[iiii][0], target=name_list[iiii][1])
         else:
                    path = None
         short.append(path)
         b_cn=np.array(b_neighbor).astype(float)
         b_jc=np.array(jc).astype(float)
         b_pa=np.array(pa).astype(float)
         b_aa=np.array(aa).astype(float)
         b_short=np.array(short).astype(float)
         where_are_NaNs = isnan(b_short)
         b_short[where_are_NaNs] = 100
         b.append(np.concatenate([b_cn,b_jc,b_pa,b_aa,b_short]))
         return np.concatenate(b)

inact=list(np.load('inact_net_symbole.npy'))
inweb=list(np.load('inweb_net_symbole.npy'))
apid=list(np.load('apid_net_symbole.npy'))
bio=list(np.load('bio_net_symbole.npy'))
bio_int=list(np.load('bio_int_net_symbole.npy'))
graphs=list(inact)+list(inweb)+list(apid)+list(bio_int)+list(bio) 
 
for iiiii in range(0,1):
    name_list=np.load('pairs_neuron_ex_10%.npy')#np.load('ziad_pairs_{}.npz'.format(iiiii))['arr_0']
    c=[]
    name_list_t=[tuple(name_list[i]) for i in range(0,len(name_list))] 
    G=nx.Graph()
    for i in range(0, len(graphs)):
            G.add_edge(graphs[i][0],graphs[i][1])    
    for iiii in range(0,len(name_list)) :
          print(iiii)  
          c.append(proximity(name_list,graphs))
    np.save('prox_neuron_ex_10%',c)
