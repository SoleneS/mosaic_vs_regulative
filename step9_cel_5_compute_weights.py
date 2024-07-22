#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:30:27 2024

@author: solene
"""



import pandas as pd
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt

path_variables='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/input/'
path_save='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/output/'

my_tissue_cel='Mus'
df_all_dists=pd.read_pickle(path_save+'df_all_dist_'+my_tissue_cel+'.pkl')
df_all_dists_cleaned=df_all_dists.loc[df_all_dists['c1']!=df_all_dists['c2']]
df_all_dists_cel=df_all_dists_cleaned



all_tissues=[0,1,2,3,4]
my_tissue_toy=0
df_all_dists=pd.read_pickle(path_save+'df_all_dist_random_'+str(my_tissue_toy)+'.pkl')
df_all_dists_cleaned=df_all_dists.loc[df_all_dists['c1']!=df_all_dists['c2']]
df_all_dists_toy=df_all_dists_cleaned


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
            r_neigh=scipy.stats.pearsonr(this_time.context_dist, this_time.exp_dist)[0]
            row_all_values={'ld':ld,'time':t,'r_context':r_neigh}
            all_values.append(row_all_values)
    df_all_values_context=pd.DataFrame(all_values)
    return df_all_values_context

def lin_corr(df_all_dists):
    all_context_dists=df_all_dists.context_dist
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
            r_lin=scipy.stats.pearsonr(this_time.lin_dist, this_time.exp_dist)[0]
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



df_all_values_context_cel=context_corr(df_all_dists_cel)
df_all_values_context_toy=context_corr(df_all_dists_toy)
stats=[]
from scipy.stats import mannwhitneyu
for t in range(50,191):
    this_time_cel=df_all_values_context_cel.loc[df_all_values_context_cel['time']==t]
    r_context_cel=this_time_cel.r_context
    r_context_cel=r_context_cel[~np.isnan(r_context_cel)]
    this_time_toy=df_all_values_context_toy.loc[df_all_values_context_toy['time']==t]
    r_context_toy=this_time_toy.r_context
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

####rearrangement
df_rearange=pd.read_pickle(path_save+'df_rearrange_'+my_tissue_cel+'.pkl')
tmp_list=[]
for t in range(50,191):
    this_time=df_rearange.loc[df_rearange['time']==t]
    mean_jac=np.mean(this_time.jac_dist)
    std_jac=np.std(this_time.jac_dist)
    row={'time':t,'mean_jac':mean_jac,'std_jac':std_jac}
    tmp_list.append(row)
df_mean_std=pd.DataFrame(tmp_list)

####divisions
nb_cells=[]
for t in range(50,191):
    this_time_dists=df_all_dists_cel.loc[df_all_dists_cel['time']==t]
    this_time_cells=this_time_dists.loc[this_time_dists['time']==t]
    cells_t=np.unique(list(this_time_cells.c1)+list(this_time_cells.c2))
    nb_cells.append({'t':t,'nb_cells':len(cells_t)})
    
df_nb_cells=pd.DataFrame(nb_cells)


#####plots
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,10),sharex=True)

ax1.scatter(df_stats_context_sig.time, df_stats_context_sig.U_diff,
            marker='o',label='context corr Udiff p<0.05', 
            facecolors='blue',
            edgecolors='blue')
ax1.scatter(df_stats_context_non_sig.time, df_stats_context_non_sig.U_diff,
            marker='o',label='context corr Udiff ns', 
            facecolors='none',
            edgecolors='blue')

ax1.scatter(df_stats_lin_sig.time, df_stats_lin_sig.U_diff,
            marker='o', label='lineage corr Udiff p<0.05',
            facecolors='red',
            edgecolors='red')
ax1.scatter(df_stats_lin_non_sig.time, df_stats_lin_non_sig.U_diff,
            marker='o', label='lineage corr Udiff ns',
            facecolors='none',
            edgecolors='red')
#ax1.set_xlabel('time')
ax1.set_ylabel('Udiff',fontsize=16)
ax1.axhline(y=-0.01,color='r',linestyle='-')
ax1.axhline(y=-0.01+0.29,color='r',linestyle='--')
ax1.axhline(y=-0.01-0.29,color='r',linestyle='--')
ax1.axhline(y=-0.09,color='b',linestyle='-')
ax1.axhline(y=-0.09+0.62,color='b',linestyle='--')
ax1.axhline(y=-0.09-0.62,color='b',linestyle='--')



ax2.errorbar(x=df_mean_std.time,y=df_mean_std.mean_jac,yerr=df_mean_std.std_jac)
ax2.set_xlabel('time',fontsize=16)
ax2.set_ylabel('rearagement rate',fontsize=16)
ax3=ax2.twinx()
ax3.plot(df_nb_cells.t, df_nb_cells.nb_cells,linestyle='-',color='black')
ax3.set_ylabel('nb cells',fontsize=16)
# plt.savefig(path_save+'legends.jpg',dpi=400)
plt.tight_layout()
plt.savefig(path_save+'divisions_rearange_'+my_tissue_cel+'.jpg', dpi=400)
plt.close()