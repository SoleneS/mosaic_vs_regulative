#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:50:22 2024

@author: solene
"""


####INPUTS:
####1)df_3D_PCA: dataframe with cell name, time, 3D positions
####2)normalized expression tables (i) df_prot_exp_n_LO.pkl for the Lineage-only model 
####3)Delaunay distances at each time point in df_dist_Dln folder
####OUTPUTS:
####folder context_dist_random with a dataframe per time point with all the context distances
        

import pandas as pd
import numpy as np
import scipy.spatial


path_variables='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/input/'
path_save='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/output/'


####3D positions and time 
df_3D_PCA=pd.read_pickle(path_variables+'df_3D_PCA_smooth.pkl')

####normalised expression 
df_exp=pd.read_pickle(path_variables+'df_prot_exp_n_LO.pkl') ####change if toy model

####find neighbours of a cell at a given time
def find_nei(cell,time):
    df_dist_Dln=pd.read_pickle(path_variables+'df_dist_Dln/'+'df_dist_Dln_time'+str(t)+'.pkl')
    line=df_dist_Dln[cell]
    neigh=list(line.loc[line==1].index)
    return neigh

####get the mean expresssion profile of the neighbours of a given cell
def mean_exp(neigh):
    df_exp_neigh=df_exp.loc[df_exp.index.isin(neigh)]
    mean_exp_nei=df_exp_neigh.mean(axis=0)
    return mean_exp_nei
    
####for each time, for each pair of cells existing at that time, record the context distance
for t in range(13,191):
    print(t)
    this_time_cells=np.unique(df_3D_PCA.loc[df_3D_PCA['time']==t].cell_name)
    dict_mean={} ####record the mean expression of its neighbours
    for c in this_time_cells:
        neigh=find_nei(c,t)
        dict_mean[c]=mean_exp(neigh)
    ###distances
    df_n=pd.DataFrame(index=this_time_cells, columns=this_time_cells)
    for ic1 in range(np.shape(this_time_cells)[0]):
        c1=this_time_cells[ic1]
        v1=dict_mean[c1]
        for ic2 in range(ic1,np.shape(this_time_cells)[0]):
            c2=this_time_cells[ic2]
            v2=dict_mean[c2]
            r=scipy.spatial.distance.correlation(v1, v2)
            df_n[c1][c2]=r
    df_n.to_pickle(path_save+'context_dist_LO/'+'df_context_dist_LO_t'+str(t)+'.pkl')

####symmetrize
for t in range(13,191):
      print(t)
      df_nei=pd.read_pickle(path_save+'context_dist_LO/'+'df_context_dist_LO_t'+str(t)+'.pkl')
      cells=df_nei.index
      for i1 in range(len(cells)):
          c1=cells[i1]
          for i2 in range(i1,len(cells)): 
              c2=cells[i2]
              df_nei[c2][c1]=df_nei[c1][c2]
      df_nei.to_pickle(path_save+'context_dist_LO/'+'df_context_dist_LO_t'+str(t)+'.pkl')