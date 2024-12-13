#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 13:52:31 2024

@author: solene
"""

import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import shortest_path
#####load complete series of cells names
path_variables='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/input/'
c=pd.read_pickle(path_variables+'c.pkl')


####load spatial coordinates
path_s='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/Li_al_3Dpositions/'
df_3D=pd.read_csv(path_s+'WT-EMB13.txt',sep='\t')####the embryo with most cells recorded

times=df_3D.time
times=pd.Series.unique(times)
times=np.array(times)
times=times.astype(int)
###################################""
def collect_edges(tri):
        edges = set()
        def sorted_tuple(a,b):
            return (a,b) if a < b else (b,a)
        # Add edges of tetrahedron (sorted so we don't add an edge twice, even if it comes in reverse order).
        for (i0, i1, i2, i3) in tri.simplices:
            edges.add(sorted_tuple(i0,i1))
            edges.add(sorted_tuple(i0,i2))
            edges.add(sorted_tuple(i0,i3))
            edges.add(sorted_tuple(i1,i2))
            edges.add(sorted_tuple(i1,i3))
            edges.add(sorted_tuple(i2,i3))
        return edges

###########################################


for i_t in range(np.shape(times)[0]):
    print(i_t)
    t=times[i_t]
    ind_t=df_3D.index[df_3D['time']==t]
    cells_t=df_3D.loc[ind_t]
    cells_t=cells_t.reset_index(drop=True)
    df_dist_Dln=pd.DataFrame(index=c,columns=c) ####store there
    N_cells=np.shape(cells_t)[0]
    if N_cells<4:
        continue
    coord_3D=cells_t[['X','Y','Z']]
    arr_3D=np.array(coord_3D)
    tri = Delaunay(arr_3D)
    edges=collect_edges(tri)
    G = np.zeros((arr_3D.shape[0],arr_3D.shape[0]))
    for e in edges:
        G[e[0],e[1]] = 1
        G[e[1],e[0]] = 1
    dist_matrix = shortest_path(G, return_predecessors=False, directed=False, unweighted=True)
    for i in range(N_cells):
        name1=cells_t.loc[i,'cell_name']
        for j in range(i,N_cells):
            name2=cells_t.loc[j,'cell_name']
            their_dist=dist_matrix[i,j]
            df_dist_Dln[name1][name2]=their_dist
            df_dist_Dln[name2][name1]=their_dist

    ####save it 
    df_dist_Dln.to_pickle(path_variables+'df_dist_Dln/df_dist_Dln_time'+str(t)+'.pkl')








    