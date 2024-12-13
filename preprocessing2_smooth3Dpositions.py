#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:32:41 2024

@author: solene
"""


import pandas as pd
import numpy as np

###lineage
path_lin="/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/4D_protein_atlas/data_malek/"
path_saved='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/DevTrajectory/variables/'
mat_adj_all=np.load(path_lin+'genealogical_distance_static_packer_alphabetique.npy',allow_pickle=True)
names_genea=np.load(path_lin+'names_genea_sort.npy', allow_pickle=True)
names_genea_liste=names_genea.tolist()
df_C = pd.DataFrame(mat_adj_all,columns=names_genea_liste, index=names_genea_liste)
df_g=pd.DataFrame(index=names_genea_liste,columns=['g'])
for i in range(np.shape(names_genea_liste)[0]):
    cell=names_genea_liste[i]
    d=df_C[cell]['P0']
    df_g.loc[cell,'g']=d

gen=np.unique(np.array(df_g.g))
gen=gen.astype(int)
df_3D_PCA=pd.read_pickle(path_saved+'df_3D_PCA.pkl')

def find_daughters(this_cell):
    his_gen=df_g.loc[this_cell,'g']
    line=df_C.loc[this_cell]
    line=line.astype(int)
    ####those who are at dist 1
    ind=line.index[line==1]
    ind=ind.values
    ind=ind.tolist()
    ###those who are at gen +1
    daughters=[]
    for c in ind:
        gen_c=df_g.loc[c,'g']
        if gen_c==his_gen+1:
            daughters.append(c)
    return daughters

def find_mother(this_cell):
    mother=[]
    his_gen=df_g.loc[this_cell,'g']
    try:
        his_gen=his_gen[0]
    except:
        his_gen=his_gen
        
    line=df_C.loc[this_cell]
    
    vals=np.array(line.values)
    vals=np.reshape(vals,(1342,1))
    line=pd.DataFrame(vals,index=df_C.index, columns=['d'])
    line=line.astype(int)
    Neigh=line.loc[line['d']==1]
    Neigh=Neigh.index
    for c in Neigh:
        gen_c=df_g.loc[c,'g']
        if gen_c==his_gen-1:
            mother.append(c)
    return mother

def all_nodes_downstream(name):
    his_gen=df_g.loc[name,'g']
    ####initiate 
    family=[name]
    working_list=[name]
    while len(working_list)>0:
        new_working_list=[]
        for this_cell in working_list:
            ###find daughters of every item of the working_list
            #####put it in the new_working_list
            daughters=find_daughters(this_cell)
            new_working_list=new_working_list+daughters
        working_list=new_working_list
        family=family+new_working_list
    if np.shape(family)[0]==1:
        print('its fine')
    return family

cs=names_genea_liste
leaves=[]
for c in cs:
    his_N=all_nodes_downstream(c)
    if np.shape(his_N)[0]==1:
        leaves.append(c)

lineages=[]
for c in leaves:
    L=[c]
    current=c
    while 1:
        his_mum=find_mother(current)
        
        if np.shape(his_mum)[0]>0:
            L.append(his_mum[0])
            current=his_mum
        else:
            break
    lineages.append(L)


#####smooth df_3D_PCA by track
import scipy.signal
####declare a list of dataframes
all_tracks=pd.DataFrame()
i=0
for lin_list in lineages:
    print(i)
    i=i+1
    df_track=pd.DataFrame()
    for cell in lin_list:
        this_cell_3D=df_3D_PCA.loc[df_3D_PCA['cell_name']==cell]
        df_track=pd.concat([df_track,this_cell_3D],axis=0)
    df_track=df_track.sort_values(by=['time'])
    df_track=df_track.reset_index(drop=True)
    xs=np.array(df_track.X)
    ys=np.array(df_track.Y)
    zs=np.array(df_track.Z)
    xs_smooth=scipy.signal.savgol_filter(xs,21,2)
    ys_smooth=scipy.signal.savgol_filter(ys,21,2)
    zs_smooth=scipy.signal.savgol_filter(zs,21,2)
    #####replace values in dataframe, store the dataframe in a list
    df_track['X']=xs_smooth
    df_track['Y']=ys_smooth
    df_track['Z']=zs_smooth
    all_tracks=pd.concat([all_tracks,df_track],axis=0)
    
all_tracks=all_tracks.drop_duplicates()


df_3D_PCA_smooth=pd.DataFrame()
i_smooth=0
all_cells=df_3D_PCA.cell_name
all_cells=np.unique(all_cells)
all_cells=list(all_cells)
i_test=0
for cell in all_cells:
    print(i_test)
    i_test=i_test+1
    this_cell_3D=all_tracks.loc[all_tracks['cell_name']==cell]
    times=this_cell_3D.time
    for t in times:
        this_cell_this_time=this_cell_3D.loc[this_cell_3D['time']==t]
        xs=np.array(this_cell_this_time.X)
        ys=np.array(this_cell_this_time.Y)
        zs=np.array(this_cell_this_time.Z)
        xmean=np.mean(xs)
        ymean=np.mean(ys)
        zmean=np.mean(zs)
        df_3D_PCA_smooth.loc[i_smooth,'time']=t
        df_3D_PCA_smooth.loc[i_smooth,'cell_name']=cell
        df_3D_PCA_smooth.loc[i_smooth,'X']=xmean
        df_3D_PCA_smooth.loc[i_smooth,'Y']=ymean
        df_3D_PCA_smooth.loc[i_smooth,'Z']=zmean
        i_smooth=i_smooth+1
        

df_3D_PCA_smooth=df_3D_PCA_smooth.drop_duplicates()
df_3D_PCA_smooth.to_pickle(path_saved+'df_3D_PCA_smooth.pkl')


