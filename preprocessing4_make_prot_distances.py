#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:45:17 2024

@author: solene
"""
import pandas as pd
import numpy as np



####import lineage data########################################################
path_variables="/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/4D_protein_atlas/data_malek/"
path_saved='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/DevTrajectory/variables/'
mat_adj_all=np.load(path_variables+'genealogical_distance_static_packer_alphabetique.npy',allow_pickle=True)
names_genea=np.load(path_variables+'names_genea_sort.npy', allow_pickle=True)
names_genea_liste=names_genea.tolist()
df_C = pd.DataFrame(mat_adj_all,columns=names_genea_liste, index=names_genea_liste)
df_g=pd.DataFrame(index=names_genea_liste,columns=['g'])
for i in range(np.shape(names_genea_liste)[0]):
    cell=names_genea_liste[i]
    d=df_C[cell]['P0']
    df_g.loc[cell,'g']=d
gen=np.unique(np.array(df_g.g))
gen=gen.astype(int)


###########load space and normalized expression data######################################
exp=pd.read_pickle(path_saved+'df_prot_exp_n.pkl')
vectors_n=np.array(exp)
c=exp.index
c_list=list(c)


####add the origin cell#######################################################
ref_cells=['ABa','ABp','P2','EMS']
N_TFs=np.shape(vectors_n)[1]
rc_exps=np.zeros((4,N_TFs))
i_emb=0
for rc in ref_cells:
    ind_rc=c_list.index(rc)
    exp_rc=vectors_n[ind_rc,:]
    rc_exps[i_emb,:]=exp_rc
    i_emb=i_emb+1
v_ref=np.mean(rc_exps,axis=0)
v_ref=v_ref.reshape((1,266))
c_list.append('O')
vectors_n=np.concatenate((vectors_n,v_ref))

#########make correlation distances in prot expression########################
import scipy.spatial
N_cells=np.shape(c_list)[0]
N_TF=np.shape(vectors_n)[1]
r_dist=np.zeros((N_cells,N_cells))
for i in range(N_cells):
    print(i)
    for j in range(i+1,N_cells):
        v1=vectors_n[i,:]
        v2=vectors_n[j,:]
        r=scipy.spatial.distance.correlation(v1,v2)
        r_dist[i,j]=r
        r_dist[j,i]=r

df_r=pd.DataFrame(r_dist, index=exp.index, columns=exp.index)
df_r.to_pickle(path_variables+'df_exp_dist.pkl')
