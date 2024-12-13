import numpy as np
import pandas as pd
import copy
import os
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
import scipy.spatial

path_variables='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/input/'
path_save='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/output'




path='/home/solene/Documents/Solene/DataMarseille/morphotranscriptimics/Disentangle/code_def/Expression_Pattern/'
files=os.listdir(path)
df_all=[]


for f in files:
    df=pd.read_csv(path+f,sep='\t')
    df_all.append(df)
    
exp=pd.concat(df_all, ignore_index=True)
exp=pd.pivot_table(exp,index='Cell')
exp=exp.fillna(0)
vectors=exp.to_numpy()
vectors_n=scaler.fit_transform(vectors)
df_exp=pd.DataFrame(vectors_n)
df_exp.index=exp.index

df_exp_shuffled=df_exp.sample(frac=1).reset_index(drop=True)
df_exp_shuffled.index=df_exp.index
df_exp_shuffled.to_pickle(path_variables+'df_prot_exp_n_random.pkl')

exp_array=np.array(df_exp_shuffled)
N=len(df_exp_shuffled)
r_dist=np.zeros((N,N))
for i in range(N):
    print(i)
    for j in range(i+1,N):
        v1=exp_array[i,:]
        v2=exp_array[j,:]
        r=scipy.spatial.distance.correlation(v1, v2)
        r_dist[i,j]=r
        r_dist[j,i]=r
df_r=pd.DataFrame(r_dist, index=df_exp_shuffled.index, columns=df_exp_shuffled.index)
df_r.to_pickle(path_variables+'df_exp_dist_random.pkl')

####etape préalable: regrouper tous les tissus df_dist2final pour tous les tissus
# ####faire des clusters avec les leaves, en choisir 1 comme faux tissu
tissues=['Neu','Ski','Mus','Pha','Int']
df_dist2final_all=pd.DataFrame()
for tis in tissues:
    df_dist2final=pd.read_pickle(path_variables+'df_dist2final_'+tis+'.pkl')
    df_dist2final['tissue']=tis
    df_dist2final_all=pd.concat([df_dist2final_all,df_dist2final])

####enlever les doublons: lines qui appartiennent à plusieurs tissus
df_cleaned=pd.DataFrame()
leaves=list(np.unique(df_dist2final_all.leaf))
for lea in leaves:
    df=df_dist2final_all.loc[df_dist2final_all['leaf']==lea]
    lines=list(np.unique(df.line))
    df_add=df.loc[(df['leaf']==lea) & (df['line']==lines[0])]
    df_cleaned=pd.concat([df_cleaned,df_add])


leaves=list(np.unique(df_dist2final_all.leaf))
df_exp_leaves=df_exp_shuffled.loc[df_exp_shuffled.index.isin(leaves)]
vectors_leaves=np.array(df_exp_leaves)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(vectors_leaves)
my_labels=kmeans.labels_
df_leaves=pd.DataFrame(columns=['cell','label'])
df_leaves.cell=df_exp_leaves.index
df_leaves.label=my_labels
for lea in leaves:
    df_cleaned.loc[df_cleaned['leaf']==lea,'tissue']=df_leaves.loc[df_leaves['cell']==lea, 'label'].reset_index(drop=True)[0]

df_sorted=pd.DataFrame()
for lab in range(5):
    df=df_cleaned.loc[df_cleaned['tissue']==lab]
    lines_ids=np.unique(df.line)
    corres=pd.DataFrame(columns=['old_id','new_id'])
    corres.old_id=lines_ids
    corres.new_id=np.arange(0,len(lines_ids),1)
    new_df=copy.deepcopy(df)
    for l in lines_ids:
        new_df.loc[df['line']==l,'line']=corres.loc[corres['old_id']==l,'new_id'].reset_index(drop='True')[0]
    df_sorted=pd.concat([df_sorted,new_df])
    
df_sorted.to_pickle(path_variables+'df_dist2final_shuffled.pkl')