import numpy as np
import itertools
import csv
from keras.models import load_model

################################################  computing co-expression features
def index_calculator(flat_name,cell_type_ind,gene_index):
    all_gene_cell_ind=[]   
    for l in range(0,len(flat_name)):                                          
         cell_expression=np.array(myfile[cell_type_ind,gene_index[l]].astype(float))
         min_exp=min(cell_expression)
         max_exp=max(cell_expression)
         range_exp=(max_exp-min_exp)/10
         if range_exp==0:
            cell_index=[0 for j in range(0,len(cell_expression))]
         else:   
          range_list=[[min_exp+(range_exp*(i-1)),min_exp+(range_exp*i)] for i in range(1,11) ]
          range_list[-1][1]=range_list[-1][1]+0.1       
          cell_index=[]
          for j in range(0,len(cell_expression)):  #number of cells
             for i in range(0,len(range_list)): # len(range_list)=10
                  if range_list[i][0]<=cell_expression[j]<=range_list[i][1]: # upper bound and lower bound
                     cell_index.append(i)
                     break           
         all_gene_cell_ind.append(cell_index) 
    ind_gene1=[]
    ind_gene2=[]
    for i in range(0,len(name)):
        ind_gene1.append(np.where(flat_name==name[i][0])[0][0]) # gene1 indexes in 500,000 pairs
        ind_gene2.append(np.where(flat_name==name[i][1])[0][0]) # gene2 indexes in 500,000 pairs
    gene1_cell_index=[all_gene_cell_ind[ind_gene1[i]] for i in range(0,len(ind_gene1))] # 500,000* 6015
    gene2_cell_index=[all_gene_cell_ind[ind_gene2[i]] for i in range(0,len(ind_gene2))] # 500,000* 6015
    
    return [gene1_cell_index,gene2_cell_index]
############################################### loading data for each cell type
cell_type='neuron_ex' 
my_gene=np.load('ziad_filtered_gene.npy',allow_pickle=True)
my_cell=np.load('filtered_imputed_cell.npy',allow_pickle=True)
myfile=np.load('Ziad_filtered_expression.npz',allow_pickle=True)['arr_0']
#cell_type_ind=np.load('{}_expression_index.npy'.format(cell_type),allow_pickle=True)
cell_type_ind=np.load('expression_index_neuron_ex_1.npy'.format(cell_type),allow_pickle=True)
cc=10
#############################################  Loading multimodal-deep learning models to compute predictionscores for each pairs of genes           
model1=load_model("{}_1_model_0".format(cell_type))
model2=load_model("{}_1_model_1".format(cell_type))
model3=load_model("{}_1_model_2".format(cell_type))
model4=load_model("{}_1_model_3".format(cell_type))
model5=load_model("{}_1_model_4".format(cell_type))
for iii in range (0,60):
    all_mat=[]
    score=[]
    score1=[]
    score2=[]
    score3=[]
    score4=[]
    name=np.load('pairs_{}.npz'.format(iii),allow_pickle=True)['arr_0'] # loading gene-pairs
    proximity=np.load('prox_{}.npy'.format(iii),allow_pickle=True) # loading proximity features
    flat_name = np.array(list(set(itertools.chain.from_iterable(name)))) # unique genes
    gene_index=[]
    for l in range(0,len(flat_name)):
      gene_index.append(list(my_gene).index(flat_name[l])) 
    indice=index_calculator(flat_name,cell_type_ind,gene_index)    
    for ii in range(0,len(name)):
        print(ii)
        mat=np.zeros((cc,cc))
        for j in range(0,len(indice[0][0])):
            mat[indice[0][ii][j],indice[1][ii][j]]=mat[indice[0][ii][j],indice[1][ii][j]]+1
        all_mat.append(np.log(mat+1)) 
    score.append(model1.predict([proximity.reshape(len(proximity),5,1,1),np.array(all_mat).reshape(len(all_mat),10,10,1)],batch_size=5000,verbose=1)[:,1])
    score1.append(model2.predict([proximity.reshape(len(proximity),5,1,1),np.array(all_mat).reshape(len(all_mat),10,10,1)],batch_size=5000,verbose=1)[:,1])
    score2.append(model3.predict([proximity.reshape(len(proximity),5,1,1),np.array(all_mat).reshape(len(all_mat),10,10,1)],batch_size=5000,verbose=1)[:,1])
    score3.append(model4.predict([proximity.reshape(len(proximity),5,1,1),np.array(all_mat).reshape(len(all_mat),10,10,1)],batch_size=5000,verbose=1)[:,1])
    score4.append(model5.predict([proximity.reshape(len(proximity),5,1,1),np.array(all_mat).reshape(len(all_mat),10,10,1)],batch_size=5000,verbose=1)[:,1])
    score_all=np.mean([score1[0],score2[0],score3[0],score4[0],score[0]],axis=0)
    np.save('{}_1_prediction_score_{}'.format(cell_type,iii),score_all)
