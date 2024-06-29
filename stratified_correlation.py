#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:01:40 2024

@author: solene
"""




import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from collections import Counter
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()



path_variables='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/variables/'
path_save='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/'

###changer le nom de datadrames en df_all_dists
all_tissues=['Neu','Mus','Ski','Pha','Int']
my_tissue_cel='Ski'
df_all_dists=pd.read_pickle(path_save+'df_all_dist_instant_celegans_'+my_tissue_cel+'.pkl')
df_all_dists_cleaned=df_all_dists.loc[df_all_dists['c1']!=df_all_dists['c2']]
df_all_dists_cel=df_all_dists_cleaned
df_all_dists_cel.rename(columns={'instant_dist': 'neigh_dist'}, inplace=True)

all_tissues=[0,1,2,3,4]
my_tissue_toy=0
df_all_dists=pd.read_pickle(path_save+'df_all_dist_shuffled_instant'+str(my_tissue_toy)+'.pkl')
df_all_dists_cleaned=df_all_dists.loc[df_all_dists['c1']!=df_all_dists['c2']]
df_all_dists_toy=df_all_dists_cleaned
df_all_dists_toy.rename(columns={'instant_dist': 'neigh_dist'}, inplace=True)


def context_corr(df_all_dists):
    all_lin_dists=np.unique(df_all_dists.lin_dist)
    all_values=[]
    for ld in all_lin_dists:
        print(ld)
        this_ld=df_all_dists.loc[df_all_dists['lin_dist']==ld]
        times=np.unique(this_ld.time)
        for t in times:
            this_time=this_ld.loc[this_ld['time']==t]
            if len(this_time)<2:
                continue
            r_neigh=scipy.stats.pearsonr(this_time.neigh_dist, this_time.corr_dist)[0]
            row_all_values={'ld':ld,'time':t,'r_neigh':r_neigh}
            all_values.append(row_all_values)
    df_all_values_context=pd.DataFrame(all_values)
    return df_all_values_context

def lin_corr(df_all_dists):
    all_context_dists=df_all_dists.neigh_dist
    num_bins=10
    bin_edges=np.linspace(min(all_context_dists), max(all_context_dists), num_bins + 1)
    binned_values=np.digitize(all_context_dists, bin_edges)
    df_all_dists['binned_context']=binned_values
    all_binned_values=np.unique(binned_values)
    all_values=[]
    for b in all_binned_values:
        print(b)
        this_bin=df_all_dists.loc[df_all_dists['binned_context']==b]
        times=np.unique(this_bin.time)
        for t in times:
            this_time=this_bin.loc[this_bin['time']==t]
            if len(this_time)<2:
                continue
            r_lin=scipy.stats.pearsonr(this_time.lin_dist, this_time.corr_dist)[0]
            row_all_values={'bin':b,'time':t,'r_lin':r_lin}
            all_values.append(row_all_values)
    df_all_values_lin=pd.DataFrame(all_values)
    return df_all_values_lin



df_all_values_lin_cel=lin_corr(df_all_dists_cel)
df_all_values_lin_toy=lin_corr(df_all_dists_toy)
stats=[]
from scipy.stats import mannwhitneyu
for t in range(50,191):
    this_time_cel=df_all_values_lin_cel.loc[df_all_values_lin_cel['time']==t]
    r_lin_cel=this_time_cel.r_lin
    r_lin_cel=r_lin_cel[~np.isnan(r_lin_cel)]
    this_time_toy=df_all_values_lin_toy.loc[df_all_values_lin_toy['time']==t]
    r_lin_toy=this_time_toy.r_lin
    r_lin_toy=r_lin_toy[~np.isnan(r_lin_toy)]
    if (len(r_lin_toy)==0) or (len(r_lin_cel)==0):
        continue
    U_cel,p=mannwhitneyu(r_lin_cel,r_lin_toy)
    nx,ny=len(r_lin_cel),len(r_lin_toy)
    U_toy=nx*ny-U_cel
    U_diff=(U_cel-U_toy)/(nx*ny)
    row={'time':t,'U_cel':U_cel,'U_toy':U_toy,'U_diff':U_diff,'p':p}
    stats.append(row)
df_stats_lin=pd.DataFrame(stats)

#plt.scatter(df_stats_lin.time, df_stats_lin.U_diff)

df_all_values_context_cel=context_corr(df_all_dists_cel)
df_all_values_context_toy=context_corr(df_all_dists_toy)
stats=[]
from scipy.stats import mannwhitneyu
for t in range(50,191):
    this_time_cel=df_all_values_context_cel.loc[df_all_values_context_cel['time']==t]
    r_context_cel=this_time_cel.r_neigh
    r_context_cel=r_context_cel[~np.isnan(r_context_cel)]
    this_time_toy=df_all_values_context_toy.loc[df_all_values_context_toy['time']==t]
    r_context_toy=this_time_toy.r_neigh
    r_context_toy=r_context_toy[~np.isnan(r_context_toy)]
    if (len(r_context_toy)==0) or (len(r_context_cel)==0):
        continue
    U_cel,p=mannwhitneyu(r_context_cel,r_context_toy)
    nx,ny=len(r_context_cel),len(r_context_toy)
    U_toy=nx*ny-U_cel
    U_diff=(U_cel-U_toy)/(nx*ny)
    row={'time':t,'U_cel':U_cel,'U_toy':U_toy,'U_diff':U_diff,'p':p}
    stats.append(row)
df_stats_context=pd.DataFrame(stats)


df_stats_context_sig=df_stats_context.loc[df_stats_context['p']<=0.05]
df_stats_context_non_sig=df_stats_context.loc[df_stats_context['p']>0.05]
df_stats_lin_sig=df_stats_lin.loc[df_stats_lin['p']<=0.05]
df_stats_lin_non_sig=df_stats_lin.loc[df_stats_lin['p']>0.05]


plt.scatter(df_stats_context_sig.time, df_stats_context_sig.U_diff,
            marker='o',label='context corr Udiff p<0.05', 
            facecolors='blue',
            edgecolors='blue',
            alpha=0.7)
plt.scatter(df_stats_context_non_sig.time, df_stats_context_non_sig.U_diff,
            marker='o',label='context corr Udiff ns', 
            facecolors='none',
            edgecolors='blue')

# plt.scatter(df_stats_lin_sig.time, df_stats_lin_sig.U_diff,
#             marker='o', label='lineage corr Udiff p<0.05',
#             facecolors='red',
#             edgecolors='red',
#             alpha=0.7)
# plt.scatter(df_stats_lin_non_sig.time, df_stats_lin_non_sig.U_diff,
#             marker='o', label='lineage corr Udiff ns',
#             facecolors='none',
#             edgecolors='red')
#plt.legend()
plt.xlabel('time')
plt.title(my_tissue_cel)
# plt.axhline(y=-0.01,color='r',linestyle='-')
# plt.axhline(y=-0.01+0.29,color='r',linestyle='--')
# plt.axhline(y=-0.01-0.29,color='r',linestyle='--')
plt.axhline(y=-0.09,color='b',linestyle='-')
plt.axhline(y=-0.09+0.62,color='b',linestyle='--')
plt.axhline(y=-0.09-0.62,color='b',linestyle='--')
plt.savefig(path_save+'context_only_diff_U_compared_to_shuffled'+my_tissue_cel+'.jpg',dpi=400)
plt.close()

# fig=go.Figure()
# fig.add_trace(go.Scatter(x=df_stats_context.time, y=df_stats_context.U_diff, mode='markers',
#                          marker=dict(color='blue')))
# fig.add_trace(go.Scatter(x=df_stats_lin.time, y=df_stats_lin.U_diff, mode='markers',
#                          marker=dict(color='red')))
# fig.add_trace(go.Scatter(x=[50,190],y=[-0.01+0.29,-0.01+0.29],mode='lines',line=dict(color='red')))
# fig.add_trace(go.Scatter(x=[50,190],y=[-0.09+0.62,-0.09+0.62],mode='lines',line=dict(color='blue')))
# fig.write_html(path_save+'diff_U_compared_to_shuffled'+my_tissue_cel+'.html')


plt.scatter(df_stats_context_sig.time, df_stats_context_sig.U_diff,
            marker='o',label='context corr Udiff p<0.05', 
            facecolors='blue',
            edgecolors='blue',
            alpha=0.7)
plt.scatter(df_stats_context_non_sig.time, df_stats_context_non_sig.U_diff,
            marker='o',label='context corr Udiff ns', 
            facecolors='none',
            edgecolors='blue')

plt.scatter(df_stats_lin_sig.time, df_stats_lin_sig.U_diff,
            marker='o', label='lineage corr Udiff p<0.05',
            facecolors='red',
            edgecolors='red',
            alpha=0.7)
plt.scatter(df_stats_lin_sig.time, df_stats_lin_sig.U_diff,
            marker='o', label='lineage corr Udiff ns',
            facecolors='none',
            edgecolors='red')
plt.ylim([-100, 100])
plt.legend()
plt.savefig(path_save+'legends.jpg',dpi=400)