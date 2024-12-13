#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:02:24 2024

@author: solene
"""



####Simplified from the core random walk with drift computation from PROSSTT algorithm Papadopoulos 2019



import numpy as np
import pandas as pd
import scipy as sp
from sklearn.preprocessing import StandardScaler
import scipy.spatial
import umap


def diffusion(steps,m):
    velocity = np.zeros((steps,m))
    walk = np.zeros((steps,m))
    walk[0,:] = np.log(sp.stats.uniform.rvs(0, 1.5,m))
    velocity[0,:] = sp.stats.norm.rvs(loc=0, scale=0.2,size=m)
    s_eps = 2 / steps
    for t in range(0, steps - 1):
        walk[t + 1,:] = walk[t,:] + velocity[t,:]
        epsilon = sp.stats.norm.rvs(loc=0, scale=s_eps, size=m)
        velocity[t + 1,:] = velocity[t,:] + epsilon
    return walk

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

def adjust_parent(walk,last_pos):
    diff=walk[0,:]-last_pos
    new_walk=walk-diff
    return new_walk


m=200
steps=3
stop_gen=13


my_path='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/input/'

mat_adj_all=np.load(my_path+'genealogical_distance_static_packer_alphabetique.npy',allow_pickle=True)
names_genea=np.load(my_path+'names_genea_sort.npy', allow_pickle=True)
names_genea_liste=names_genea.tolist()
df_C = pd.DataFrame(mat_adj_all,columns=names_genea_liste, index=names_genea_liste)
df_g=pd.DataFrame(index=names_genea_liste,columns=['g'])
for i in range(np.shape(names_genea_liste)[0]):
    cell=names_genea_liste[i]
    d=df_C[cell]['P0']
    df_g.loc[cell,'g']=d



###my_exp is a dict of arrays

def make_expression(my_exp,m):
    ####make P0
    walk_P0=diffusion(steps,m)
    my_exp['P0']=walk_P0
    current=['P0']
    #####store it###########################
    #####bfs search
    while len(current)>0:
        new_current=[]
        for c in current:
            if c=='ABprpppaaaa':
                continue
            dau=find_daughters(c)
            if len(dau)<2:
                continue
            d1=dau[0]
            d2=dau[1]
            last_position_c=my_exp[c][-1,:]
            walk_1=diffusion(steps,m)
            walk_1_adjusted=adjust_parent(walk_1,last_position_c)
            walk_2=diffusion(steps,m)
            walk_2_adjusted=adjust_parent(walk_2,last_position_c)
            my_exp[d1]=walk_1_adjusted
            my_exp[d2]=walk_2_adjusted
            new_current.append(d1)
            new_current.append(d2)
        current=new_current
    return my_exp
                



my_exp={}
my_exp=make_expression(my_exp, m)



cells_list=[]
cells_kept=list(df_g.loc[df_g['g']<=stop_gen].index)
if 'ABprpppaaaaa' in cells_kept:
    cells_kept.remove('ABprpppaaaaa')

exp_array=np.array([[0]*m]*len(cells_kept))
exp_array=exp_array.astype(float)
i=0
for key in my_exp:
    if key in cells_kept:
        cells_list.append(key)
        exp_c=my_exp[key]
        exp_c_mean=np.mean(exp_c,axis=0).reshape(1,-1)
        exp_array[i,:]=exp_c_mean
        i+=1

scaler=StandardScaler()
exp_norm=scaler.fit_transform(exp_array)

df_exp=pd.DataFrame(exp_norm, index=cells_list)
path_variables='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/variables/'
df_exp.to_pickle(path_variables+'df_exp_n_LO.pkl')


####evaluate
reducer=umap.UMAP(n_components=3)
X_t=reducer.fit_transform(exp_norm)




df_umap=pd.DataFrame(columns=['cell','x_umap','y_umap','z_umap','g'])
df_umap.cell=cells_list
df_umap.x_umap=X_t[:,0]
df_umap.y_umap=X_t[:,1]
df_umap.z_umap=X_t[:,2]
all_cells=list(df_umap.cell)
for c in all_cells:
    his_g=df_g.loc[df_g.index==c,'g'][0]
    df_umap.loc[df_umap['cell']==c,'g']=his_g



import plotly.graph_objects as go
fig=go.Figure()
fig.add_trace(go.Scatter3d( x=df_umap.x_umap, y=df_umap.y_umap, z=df_umap.z_umap, text=df_umap.cell, mode='markers',
                          marker=dict(color=df_umap.g,size=7)))
fig.write_html(my_path+'umap_LO_model.html')
fig.write_image(my_path+'umap_LO_model.jpg')

N=len(cells_list)
r_dist=np.zeros((N,N))
for i in range(N):
    print(i)
    for j in range(i+1,N):
        v1=exp_norm[i,:]
        v2=exp_norm[j,:]
        r=scipy.spatial.distance.correlation(v1, v2)
        r_dist[i,j]=r
        r_dist[j,i]=r
df_r=pd.DataFrame(r_dist, index=cells_list, columns=cells_list)

df_r.to_pickle(path_variables+'df_exp_dist_LO.pkl')

gens=np.unique(df_g.g)
tmp_list=[]
for g in range(1,stop_gen+1):
    cells_g=list(df_g.loc[df_g['g']==g].index)
    if 'ABprpppaaaaa' in cells_g:
        cells_g.remove('ABprpppaaaaa')
    for i in range(len(cells_g)):
        cell1=cells_g[i]
        for j in range(i+1,len(cells_g)):
            cell2=cells_g[j]
            r_rw=df_r[cell1][cell2]
            r_lin=df_C[cell1][cell2]
            row={'g':g,'c1':cell1,'c2':cell2,'r_rw':r_rw,'r_lin':r_lin}
            tmp_list.append(row)
df_compare=pd.DataFrame(tmp_list)
   
    

###make correlation
df_pearson=pd.DataFrame(columns=['g','r_pearson'])
for g in range(2,stop_gen):
    this_gen=df_compare.loc[df_compare['g']==g]
    fig=go.Figure()
    fig.add_trace(go.Scatter( x=this_gen.r_lin, y=this_gen.r_rw, mode='markers',
                              marker=dict(color='red',size=7)))
    fig.write_image(my_path+'lin_vs_rw_g'+str(g)+'.jpg')
    r_pearson=scipy.stats.pearsonr(this_gen.r_lin, this_gen.r_rw)[0]
    row={'g':g,'r_pearson':r_pearson}
    df_pearson.loc[len(df_pearson)]=row
    
fig=go.Figure()
fig.add_trace(go.Scatter( x=df_pearson.g, y=df_pearson.r_pearson, mode='markers',
                          marker=dict(color='red',size=7)))
fig.write_image(my_path+'lin_vs_exp_LO_pearson_m_'+str(m)+'.jpg')
