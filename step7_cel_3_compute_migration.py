#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:38:16 2024

@author: solene
"""
####INPUT
####1)lineage info genealogical_distance_static_packer_alphabetique.npy and 'names_genea_sort.npy'
####2)time and 3D positions of all cells
####3)all cells that lead to each tissue in bundles
####4) Delaunay distances between cells 'df_dist_Dln/'+'df_dist_Dln_time'+str(t)+'.pkl')
####OUTPUT
####'df_migration_'+my_tissue+'.pkl'

import pandas as pd
import numpy as np
import random

path_variables='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/input/'
path_save='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/output/'



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


df_3D_PCA=pd.read_pickle(path_variables+'df_3D_PCA_smooth.pkl')

#####calculate for each time the mean cell size (mean dist between two neighbouring cells)
mean_sizes=[]
for t in range(13,191):
    print(t)
    df_Dln=pd.read_pickle(path_variables+'df_dist_Dln/'+'df_dist_Dln_time'+str(t)+'.pkl')
    row_index, col_index = (df_Dln == 1).values.nonzero()
    all_d=[]
    my_indices=np.arange(0,len(row_index),1)
    if len(row_index)>200:
        my_indices=random.sample(range(1,len(row_index)),200)
    #####if there are more than 100, pick only randomly 100
    for i in my_indices:
        ic1=row_index[i]
        ic2=col_index[i]
        if row_index[i]<col_index[i]:
            c1=df_Dln.index[ic1]
            c2=df_Dln.index[ic2]
            pos_c1=df_3D_PCA.loc[(df_3D_PCA['time']==t) & (df_3D_PCA['cell_name']==c1)]
            pos_c1=np.array(pos_c1[['X','Y','Z']])
            pos_c2=df_3D_PCA.loc[(df_3D_PCA['time']==t) & (df_3D_PCA['cell_name']==c2)]
            pos_c2=np.array(pos_c2[['X','Y','Z']])
            d=np.linalg.norm(pos_c1-pos_c2)
            all_d.append(d)
    mean_d=np.mean(all_d)
    row={'time':t, 'mean_size':mean_d}
    mean_sizes.append(row)
df_mean_size=pd.DataFrame(mean_sizes)



all_tissues=['Neu','Ski','Mus','Pha','Int']
for my_tissue in all_tissues:
    df_bundles=pd.read_pickle(path_variables+'bundles/'+'df_bundles_'+my_tissue+'.pkl')
    cells=np.unique(df_bundles.cell)
    list_all_dist=[]
    for t in range(14,191):
        print(t)
        mean_size_t=df_mean_size.loc[df_mean_size['time']==t].mean_size.reset_index(drop=True)[0]
        ####cells_t:cells present at that time
        all_cells_t=set(df_3D_PCA.loc[df_3D_PCA['time']==t].cell_name)
        cells_set=set(cells)
        cells_t=list(all_cells_t.intersection(cells_set))
        for c in cells_t:
            previous_t=df_3D_PCA.loc[(df_3D_PCA['cell_name']==c) & (df_3D_PCA['time']==t-1)]
            if len(previous_t)>0:
                previous_t=previous_t.reset_index(drop=True)
                current_t=df_3D_PCA.loc[(df_3D_PCA['cell_name']==c) & (df_3D_PCA['time']==t)]
                current_t=current_t.reset_index(drop=True)
                previous_pos=np.array(previous_t[['X','Y','Z']])
                current_pos=np.array(current_t[['X','Y','Z']])
                dist=np.linalg.norm(current_pos-previous_pos)
                dist_norm=dist/mean_size_t
                row={'time':t,'c':c,'dist_migration':dist, 'dist_migration_norm':dist_norm}
                list_all_dist.append(row)
    df_migration=pd.DataFrame(list_all_dist)
    df_migration.to_pickle(path_save+'df_migration_'+my_tissue+'.pkl')
            
            
           

