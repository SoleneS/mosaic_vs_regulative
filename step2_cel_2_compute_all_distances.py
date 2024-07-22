#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:01:15 2024

@author: solene
"""
####INPUT
####1)lineage info genealogical_distance_static_packer_alphabetique.npy and 'names_genea_sort.npy'
####2)protein expression distances df_exp_dist.pkl
####3)time and 3D positions for each cell df_3D_PCA_smooth.pkl
####4)all cells that lead to each tissue in bundles
####5) Delaunay distances between cells 'df_dist_Dln/'+'df_dist_Dln_time'+str(t)+'.pkl')
####6) previously computed context distances context_dist/'+'df_context_dist_t'+str(t)+'.pkl'
####OUTPUT
####1) table with all distances for all pairs in a given tissue 'df_all_dist_'+my_tissue+'.pkl'
####2) table with all distances for all pairs in the whole embryo 'df_all_dist_all_tissues_pooled.pkl'


import pandas as pd
import numpy as np

path_variables='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/input/'
path_save='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/output/'


####load lineage information
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

####load protein expression distances
df_r=pd.read_pickle(path_variables+'df_exp_dist.pkl') ####change if toy model

####load time and spatial positions
df_3D_PCA=pd.read_pickle(path_variables+'df_3D_PCA_smooth.pkl')


all_tissues=['Neu','Ski','Mus','Pha','Int']
for my_tissue in all_tissues:
    df_bundles=pd.read_pickle(path_variables+'bundles/'+'df_bundles_'+my_tissue+'.pkl')
    cells=np.unique(df_bundles.cell)
    list_all_dist=[]
    for t in range(13,191):
        print(t)
        df_Dln=pd.read_pickle(path_variables+'df_dist_Dln/'+'df_dist_Dln_time'+str(t)+'.pkl')
        ####cells_t:cells present at that time
        all_cells_t=set(df_3D_PCA.loc[df_3D_PCA['time']==t].cell_name)
        cells_set=set(cells)
        cells_t=list(all_cells_t.intersection(cells_set))
        df_nei=pd.read_pickle(path_save+'context_dist/'+'df_context_dist_t'+str(t)+'.pkl')
        for ic1 in range(np.shape(cells_t)[0]):
            c1=cells_t[ic1]
            for ic2 in range(ic1,np.shape(cells_t)[0]):
                c2=cells_t[ic2]
                row={'time':t,'c1':c1,'c2':c2,'lin_dist':df_C[c1][c2],'context_dist':df_nei[c1][c2],'exp_dist':df_r[c1][c2],'Dln_dist':df_Dln[c1][c2]}
                list_all_dist.append(row)
    df_all_dist=pd.DataFrame(list_all_dist) 
    df_all_dist.to_pickle(path_save+'df_all_dist_'+my_tissue+'.pkl')



####pooled (instead of tissue per tissue)
list_all_dist=[]
for t in range(13,191):
    print(t)
    df_Dln=pd.read_pickle(path_variables+'df_dist_Dln/'+'df_dist_Dln_time'+str(t)+'.pkl')
    ####cells_t:cells present at that time
    all_cells_t=list(df_3D_PCA.loc[df_3D_PCA['time']==t].cell_name)
    df_nei=pd.read_pickle(path_save+'context_dist/'+'df_context_dist_t'+str(t)+'.pkl')
    for ic1 in range(np.shape(all_cells_t)[0]):
        c1=all_cells_t[ic1]
        for ic2 in range(ic1+1,np.shape(all_cells_t)[0]):
            c2=all_cells_t[ic2]
            row={'time':t,'c1':c1,'c2':c2,'lin_dist':df_C[c1][c2],'context_dist':df_nei[c1][c2],'exp_dist':df_r[c1][c2],'Dln_dist':df_Dln[c1][c2]}
            list_all_dist.append(row)
df_all_dist=pd.DataFrame(list_all_dist) 
df_all_dist.to_pickle(path_save+'df_all_dist_all_tissues_pooled.pkl')


