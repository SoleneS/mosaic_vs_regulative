#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 13:40:34 2024

@author: solene
"""


import os
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()



#######load the data (data with no time steps)################################
path='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/Expression_Pattern/'
path_saved='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/input/'
files=os.listdir(path)
df_all=[]

for f in files:
    df=pd.read_csv(path+f,sep='\t')
    df_all.append(df)
    
exp=pd.concat(df_all, ignore_index=True)
exp=pd.pivot_table(exp,index='Cell')
exp=exp.fillna(0)
exp=exp.reset_index()
##############################################################################


#####clean up the redundant reporters#########################################
TFs_labels=exp.columns
N_labels=np.shape(TFs_labels)[0]
TFs=pd.DataFrame(columns=['TF','strain','full'])
i_tf=0
for i in range(N_labels):
    name=TFs_labels[i]
    if '_' in name:
        pos=name.find('_')
        TF=name[0:pos]
        strain=name[pos+1:len(name)]
        TFs.loc[i_tf,'TF']=TF
        TFs.loc[i_tf,'strain']=strain
        TFs.loc[i_tf,'full']=name
        i_tf=i_tf+1

new_exp=pd.DataFrame()
new_exp['Cell']=exp.Cell
TFs_uni=np.unique(TFs.TF)
TFs_uni=list(TFs_uni)
for tf in TFs_uni:
    his_lines=TFs.loc[TFs['TF']==tf]
    his_lines=his_lines.reset_index(drop=True)
    if np.shape(his_lines)[0]==1:
        full_name=his_lines.full
       
        new_exp[tf]=exp[full_name]
    else:
        print(tf)
        his_strain1=his_lines.full[0]
        his_strain2=his_lines.full[1]
        his_col1=exp[his_strain1]
        his_col2=exp[his_strain2]
        his_cols=pd.concat([his_col1,his_col2],axis=1)
        arr=np.array(his_cols)
        mean_arr=np.mean(arr,1)
        new_exp[tf]=mean_arr
exp=new_exp
##############################################################################
c=exp.Cell
TFs=list(exp.columns)
TFs.remove('Cell')
vectors=exp.to_numpy()
vectors=vectors[:,1:np.shape(vectors)[1]]
vectors_n=scaler.fit_transform(vectors)
exp_n=pd.DataFrame(vectors_n,index=c,columns=TFs)
exp_n.reset_index()


exp_n.to_pickle(path_saved+'df_prot_exp_n.pkl')