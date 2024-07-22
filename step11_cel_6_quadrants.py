#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 18:41:01 2024

@author: solene
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path_variables='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/input/'
path_save='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/output/'

df_all_dists_pooled=pd.read_pickle(path_save+'df_all_dist_all_tissues_pooled.pkl')
df_all_dists_cleaned=df_all_dists_pooled.loc[df_all_dists_pooled['c1']!=df_all_dists_pooled['c2']]
df_all_dists_all=df_all_dists_cleaned



def make_bins(this_time,t):
    this_time_all=df_all_dists_all.loc[df_all_dists_all['time']==t]
    min_lin_dist=np.min(this_time_all.lin_dist)
    max_lin_dist=np.max(this_time_all.lin_dist)
    min_Dln_dist=np.min(this_time_all.Dln_dist)
    max_Dln_dist=np.max(this_time_all.Dln_dist)
    min_context_dist=np.min(this_time_all.context_dist)
    max_context_dist=np.max(this_time_all.context_dist)
    mid_lin=(min_lin_dist+max_lin_dist)/2
    if mid_lin==np.ceil(mid_lin):
        mid_lin=mid_lin+0.5
    mid_Dln=(min_Dln_dist+max_Dln_dist)/2
    if mid_Dln==np.ceil(mid_Dln):
        mid_Dln=mid_Dln+0.5
    #mid_Dln=2.5
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
    ###add empty column
    this_time['bin']=''
    this_time['bin']=this_time['bin'].astype(str)
    for i in range(len(this_time)):
        #print('i '+str(i))
        for i_bin, bin_coords in enumerate(bins):
            #print('i_bin '+str(i_bin))
            if(bin_coords[0][0] <= this_time.lin_dist[i] <= bin_coords[0][1] and
               bin_coords[1][0] <= this_time.Dln_dist[i] <= bin_coords[1][1] and
               bin_coords[2][0] <= this_time.context_dist[i] <= bin_coords[2][1]):
               this_time.loc[i,'bin']=bins_labels[i_bin]
    return this_time

df_binned=pd.DataFrame()
for t in range(50,190):
    print(t)
    this_time=df_all_dists_all.loc[df_all_dists_all['time']==t].reset_index(drop=True)
    this_time_binned=make_bins(this_time,t)
    df_binned=pd.concat([df_binned,this_time_binned])
    


# tmp_list=[]
# bins_labels=['111','110','101','100','011','010','001','000']
# for b in bins_labels:
#     this_b=df_binned.loc[df_binned['bin']==b]
#     row={'bin':b, 'N':(len(this_b)/len(df_binned))*100, 'mean_exp':np.mean(this_b.corr_dist), 
#          'std_exp':np.std(this_b.corr_dist)}
#     tmp_list.append(row)
# df_count=pd.DataFrame(tmp_list)

tmp_list=[]
bins_labels=['111','110','101','100','011','010','001','000']
for b in bins_labels:
    print(b)
    for t in range(50,190):
        this_b_t=df_binned.loc[(df_binned['bin']==b) & (df_binned['time']==t)]
        this_t=df_binned.loc[df_binned['time']==t]
        row={'bin':b,'time':t, 'N':(len(this_b_t)/len(this_t))*100, 'mean_exp':np.mean(this_b_t.exp_dist), 
         'std_exp':np.std(this_b_t.exp_dist)}
        tmp_list.append(row)
df_count=pd.DataFrame(tmp_list)


###chose what bins to exclude because few cells, then make plot with only the existing quadrants
bins_labels=['111','011','010','001','000']

for b in bins_labels:
    this_b=df_count.loc[df_count['bin']==b]
    plt.errorbar(x=this_b.time, y=this_b.N,label=b)

plt.legend()
plt.xlabel('time',fontsize=16)
plt.ylabel('Proportion of cells',fontsize=16)
plt.title('all_tissues_pooled',fontsize=16)
plt.savefig(path_save+'N_quandrants.jpg', dpi=300)

plt.close()
for b in bins_labels:
    this_b=df_count.loc[df_count['bin']==b]
    #plt.errorbar(x=this_b.time, y=this_b.mean_exp, yerr=this_b.std_exp,alpha=0.5,label=b)
    plt.scatter(x=this_b.time, y=this_b.mean_exp,alpha=0.5,label=b)
plt.legend()
plt.xlabel('time',fontsize=16)
plt.ylabel('mean exp dist',fontsize=16)
plt.title('all_tissues_pooled',fontsize=16)
plt.savefig(path_save+'exp_quadrants.jpg', dpi=300)

