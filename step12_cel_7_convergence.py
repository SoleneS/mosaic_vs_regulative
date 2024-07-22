#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:54:36 2024

@author: solene
"""




import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

my_tissue='Ski'

path_variables='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/input/'
path_save='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/output/'

df_bundles=pd.read_pickle(path_variables+'bundles/'+'df_bundles_'+my_tissue+'.pkl')


path_lin="/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/4D_protein_atlas/data_malek/"
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

####draw in lineage
####lineage tree positions
lin_coord_clean=pd.read_pickle(path_variables+'linear_tree_position.pkl')
df_draw_all=pd.read_pickle(path_variables+'linear_tree_draw.pkl')
df_vert=pd.read_pickle(path_variables+'vertical_edges.pkl')

cells=list(np.unique(df_draw_all.cell))

c1='ABplaapppp'
c2='Cpappd'
b1=df_bundles.loc[df_bundles['leaf']==c1].reset_index(drop=True)
b2=df_bundles.loc[df_bundles['leaf']==c2].reset_index(drop=True)

####draw whole lineage tree
fig=go.Figure()
for c in cells:
    this_cell=df_draw_all[df_draw_all['cell']==c]
    fig.add_trace(go.Scatter( x =this_cell.x, y= this_cell.y, text=this_cell.cell, mode='lines', line=dict(color='grey')))
vert_edges=list(np.unique(df_vert.e))
for e in vert_edges:
    this_edge=df_vert[df_vert['e']==e]
    fig.add_trace(go.Scatter(x=this_edge.x, y=this_edge.y, mode='lines',line=dict(color='grey')))
###############################"
his_cells1=b1.cell
his_cells1=list(his_cells1)+['AB','P0']
his_cells2=b2.cell
his_cells2=list(his_cells2)+['P1','P0']
for c in his_cells1:
    this_cell=df_draw_all[df_draw_all['cell']==c]
    fig.add_trace(go.Scatter( x =this_cell.x, y= this_cell.y, text=this_cell.cell, mode='lines', line=dict(color='green')))
for c in his_cells2:
    this_cell=df_draw_all[df_draw_all['cell']==c]
    fig.add_trace(go.Scatter( x =this_cell.x, y= this_cell.y, text=this_cell.cell, mode='lines', line=dict(color='red')))
fig.update_layout(font=dict(size=30),showlegend=False,yaxis=dict(showticklabels=False),xaxis=dict(title='time'))
fig.write_image(path_save+'lineage_particular_case'+'.jpg')

####load all the variables
####umap

embed=pd.read_pickle(path_variables+'umap_coord.pkl')     
fig=go.Figure()
fig.add_trace(go.Scatter(x=embed.umap_x, y=embed.umap_y,mode='markers',marker=dict(size=7,color='grey',opacity=0.2)))
his_cells1_xy=[]
his_cells2_xy=[]
for c in his_cells1:
    if c in ['P0','AB','P1']:
        continue
    x=embed.loc[embed['cell']==c,'umap_x'].reset_index(drop=True)[0]
    y=embed.loc[embed['cell']==c,'umap_y'].reset_index(drop=True)[0]
    his_cells1_xy.append([x,y])
his_cells1_xy=np.array(his_cells1_xy)
fig.add_trace(go.Scatter(x=his_cells1_xy[:,0], y=his_cells1_xy[:,1],text=his_cells1,mode='lines',line=dict(color='green')))
for c in his_cells2:
    if c in ['P0','AB','P1']:
        continue
    x=embed.loc[embed['cell']==c,'umap_x'].reset_index(drop=True)[0]
    y=embed.loc[embed['cell']==c,'umap_y'].reset_index(drop=True)[0]
    his_cells2_xy.append([x,y])
his_cells2_xy=np.array(his_cells2_xy)
fig.add_trace(go.Scatter(x=his_cells2_xy[:,0], y=his_cells2_xy[:,1],text=his_cells2,mode='lines',line=dict(color='red')))
fig.update_layout(showlegend=False, xaxis=dict(showticklabels=False),yaxis=dict(showticklabels=False))
fig.write_image(path_save+'umap_particular_case.jpg')

#####tracks in space
df_3D_PCA=pd.read_pickle(path_variables+'df_3D_PCA_smooth.pkl')

fig=go.Figure()
fig.add_trace(go.Scatter3d(x=df_3D_PCA.X, y=df_3D_PCA.Y, z=df_3D_PCA.Z, mode='markers', marker=dict(color='grey',opacity=0.2,size=1)))

df_this_line=pd.DataFrame(columns=['time','cell_name','X','Y','Z'])
for c in reversed(his_cells1):
    this_cell=df_3D_PCA.loc[df_3D_PCA['cell_name']==c]
    df_this_line=pd.concat([df_this_line,this_cell],axis=0)
fig.add_trace(go.Scatter3d(x=df_this_line.X, y=df_this_line.Y, z=df_this_line.Z, mode='lines', 
                           line=dict(color='green',width=3),
              showlegend=False))

df_this_line=pd.DataFrame(columns=['time','cell_name','X','Y','Z'])
for c in reversed(his_cells2):
    this_cell=df_3D_PCA.loc[df_3D_PCA['cell_name']==c]
    df_this_line=pd.concat([df_this_line,this_cell],axis=0)
fig.add_trace(go.Scatter3d(x=df_this_line.X, y=df_this_line.Y, z=df_this_line.Z, mode='lines', 
                           line=dict(color='red',width=3),
                           showlegend=False))
fig.update_layout(
    scene=dict(
       xaxis=dict(showticklabels=False,title='X'),
       yaxis=dict(showticklabels=False,title='Y'),
       zaxis=dict(showticklabels=False,title='Z')))
fig.write_image(path_save+'space_particular_case.jpg')

df_r=pd.read_pickle(path_variables+'df_exp_dist.pkl')


path_lin="/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/4D_protein_atlas/data_malek/"
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


df_all_dists_pooled=pd.read_pickle(path_save+'df_all_dist_all_tissues_pooled.pkl')
df_all_dists_cleaned=df_all_dists_pooled.loc[df_all_dists_pooled['c1']!=df_all_dists_pooled['c2']]
df_all_dists_all=df_all_dists_cleaned


def make_bins(cell1,cell2,t):
    this_time_all=df_all_dists_all.loc[df_all_dists_all['time']==t]
    min_lin_dist=np.min(this_time_all.lin_dist)
    max_lin_dist=np.max(this_time_all.lin_dist)
    min_Dln_dist=np.min(this_time_all.Dln_dist)
    max_Dln_dist=np.max(this_time_all.Dln_dist)
    min_context_dist=np.min(this_time_all.context_dist)
    max_context_dist=np.max(this_time_all.context_dist)
    mid_lin=(min_lin_dist+max_lin_dist)/2
    mid_Dln=(min_Dln_dist+max_Dln_dist)/2
    if mid_lin==np.ceil(mid_lin):
        mid_lin=mid_lin+0.5
    mid_Dln=(min_Dln_dist+max_Dln_dist)/2
    if mid_Dln==np.ceil(mid_Dln):
        mid_Dln=mid_Dln+0.5
    mid_context=(min_context_dist+max_context_dist)/2
    bins_labels=['111','110','101','100','011','010','001','000']
    bins=[
        ((min_lin_dist,mid_lin), (min_Dln_dist,mid_Dln), (min_context_dist,mid_context)),
        ((min_lin_dist,mid_lin), (min_Dln_dist,mid_Dln), (mid_context,max_context_dist)),
        ((min_lin_dist,mid_lin), (mid_Dln,max_Dln_dist), (min_context_dist,mid_context)),
        ((min_lin_dist,mid_lin), (mid_Dln,max_Dln_dist), (mid_context,max_context_dist)),
        ((mid_lin,max_lin_dist), (min_Dln_dist,mid_Dln), (min_context_dist,mid_context)),
        ((mid_lin, max_lin_dist), (min_Dln_dist,mid_Dln),(mid_context,max_context_dist)),
        ((mid_lin,max_lin_dist), (mid_Dln,max_Dln_dist), (min_context_dist,mid_context)),
         ((mid_lin,max_lin_dist),(mid_Dln,max_Dln_dist),(mid_context,max_context_dist))
        ]
    ###
    my_pair=this_time_all.loc[(this_time_all['c1']==cell1) & (this_time_all['c2']==cell2)]
    if len(my_pair)==0:
        my_pair=this_time_all.loc[(this_time_all['c1']==cell2) & (this_time_all['c2']==cell1)]
    my_pair=my_pair.reset_index(drop=True)
    his_bin=[]
    for i_bin, bin_coords in enumerate(bins):
        #print('i_bin '+str(i_bin))
        if(bin_coords[0][0] <= my_pair.lin_dist[0] <= bin_coords[0][1] and
           bin_coords[1][0] <= my_pair.Dln_dist[0] <= bin_coords[1][1] and
           bin_coords[2][0] <= my_pair.context_dist[0] <= bin_coords[2][1]):
           his_bin=bins_labels[i_bin]
    return his_bin



list_all_dist=[]
# for t in range(1,13):
#     print(t)
#     cell1='P0'
#     cell2='AB'
#     lin_dist=df_C[cell1][cell2]
#     context_dist=df_r[cell1][cell2]
#     corr_dist=df_r[cell1][cell2]
#     Dln_dist=1
#     norm_corr_dist=1
#     norm_context_dist=1
#     his_bin=make_bins(cell1,cell2)

for t in range(13,191):
    print(t)
    pooled_t=df_all_dists_all.loc[df_all_dists_all['time']==t]
    pooled_t_mean_exp=np.mean(pooled_t.exp_dist)
    pooled_t_mean_context=np.mean(pooled_t.context_dist)
    df_Dln=pd.read_pickle(path_variables+'df_dist_Dln/'+'df_dist_Dln_time'+str(t)+'.pkl')
    ####cells_t:cells present at that time
    all_cells_t=set(df_3D_PCA.loc[df_3D_PCA['time']==t].cell_name)
    cell1=all_cells_t.intersection(his_cells1).pop()
    cell2=all_cells_t.intersection(his_cells2).pop()
    his_bin=make_bins(cell1,cell2,t)
    df_nei=pd.read_pickle(path_save+'context_dist/'+'df_context_dist_t'+str(t)+'.pkl')
    row={'time':t,'c1':cell1,'c2':cell2,'lin_dist':df_C[cell1][cell2],'context_dist':df_nei[cell1][cell2],'corr_dist':df_r[cell1][cell2],
         'Dln_dist':df_Dln[cell1][cell2], 'norm_exp_dist':df_r[cell1][cell2]/pooled_t_mean_exp,
         'norm_context_dist':df_nei[cell1][cell2]/pooled_t_mean_context,
         'bin':his_bin}
    #df_all_dist.loc[len(df_all_dist)]=row
    list_all_dist.append(row)
df_all_dist_conv=pd.DataFrame(list_all_dist) 

plt.plot(df_all_dist_conv.time,df_all_dist_conv.norm_exp_dist,linestyle='-',color='black')
bins=['111','011','010','001','000']
for my_bin in bins:
    df_bin=df_all_dist_conv.loc[df_all_dist_conv['bin']==my_bin]
    plt.scatter(df_bin.time,df_bin.norm_exp_dist,label=my_bin,alpha=0.5)
    plt.xlim([50, 190])
plt.xlabel('time',fontsize=16)
plt.ylabel('normalised \nexpression distance',fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig(path_save+'particular_case_exp.jpg',dpi=300)

plt.close()


plt.plot(df_all_dist_conv.time,df_all_dist_conv.norm_context_dist,linestyle='-',color='black')
for my_bin in bins:
    df_bin=df_all_dist_conv.loc[df_all_dist_conv['bin']==my_bin]
    plt.scatter(df_bin.time,df_bin.norm_context_dist,label=my_bin,alpha=0.5)
plt.xlim([50, 190])
plt.xlabel('time',fontsize=16)
plt.ylabel('normalised \ncontext distance',fontsize=16)
plt.tight_layout()
plt.savefig(path_save+'particular_case_context.jpg',dpi=300)

plt.close()

plt.plot(df_all_dist_conv.time,df_all_dist_conv.Dln_dist,linestyle='-',color='black')
for my_bin in bins:
    df_bin=df_all_dist_conv.loc[df_all_dist_conv['bin']==my_bin]
    plt.scatter(df_bin.time,df_bin.Dln_dist,label=my_bin,alpha=0.5)
plt.xlim([50, 190])
plt.xlabel('time',fontsize=16)
plt.ylabel(' \nphysical distance',fontsize=16)
plt.tight_layout()
plt.savefig(path_save+'particular_case_Dln.jpg',dpi=300)

plt.close()