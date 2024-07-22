#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:48:56 2024

@author: solene
"""
####INPUT
####1)lineage info genealogical_distance_static_packer_alphabetique.npy and 'names_genea_sort.npy'
####2)time and 3D positions of all cells
####3)all cells that lead to each tissue in bundles
####4) Delaunay distances between cells 'df_dist_Dln/'+'df_dist_Dln_time'+str(t)+'.pkl')
####OUTPUT
####'df_rearrange_'+my_tissue+'.pkl'

import pandas as pd
import numpy as np

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


all_tissues=['Neu','Ski','Mus','Pha','Int']
for my_tissue in all_tissues:
    df_bundles=pd.read_pickle(path_variables+'bundles/'+'df_bundles_'+my_tissue+'.pkl')
    cells=np.unique(df_bundles.cell)
    list_all_dist=[]
    for t in range(14,191):
        print(t)
        df_Dln_previous=pd.read_pickle(path_variables+'df_dist_Dln/'+'df_dist_Dln_time'+str(t-1)+'.pkl')
        df_Dln=pd.read_pickle(path_variables+'df_dist_Dln/'+'df_dist_Dln_time'+str(t)+'.pkl')
        ####cells_t:cells present at that time
        all_cells_t=set(df_3D_PCA.loc[df_3D_PCA['time']==t].cell_name)
        cells_set=set(cells)
        cells_t=list(all_cells_t.intersection(cells_set))
        for c in cells_t:
            ####current_neighbours
            neigh_t=df_Dln.loc[df_Dln.index==c].reset_index(drop=True)
            neigh_t=list(neigh_t.columns[neigh_t.iloc[0]==1])
            neigh_t=set(neigh_t)
            ####previous neighbours
            neigh_previous=df_Dln_previous.loc[df_Dln_previous.index==c].reset_index(drop=True)
            neigh_previous=list(neigh_previous.columns[neigh_previous.iloc[0]==1])
            neigh_previous=set(neigh_previous)
            #####jaccard distance
            inter=neigh_t.intersection(neigh_previous)
            union=neigh_t.union(neigh_previous)
            j=1-(len(inter)/len(union))
            row={'time':t,'c':c,'jac_dist':j}
            list_all_dist.append(row)
    df_rearrange=pd.DataFrame(list_all_dist)
    df_rearrange.to_pickle(path_save+'df_rearrange_'+my_tissue+'.pkl')
                
            
           

