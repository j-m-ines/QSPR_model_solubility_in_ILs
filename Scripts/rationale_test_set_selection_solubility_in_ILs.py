import pandas as pd
import numpy as np
from sklearn import metrics
import re
from scipy.spatial import distance

#change path to your directory
path1='Documents/SLE/new_modeling/new_dataset_chemaxon_full.txt'

#read full dataset
data = pd.read_csv(path1, sep="\t")

#convert solubility to three classes
data['Classification'] = pd.cut(data['Parts of Solvent'], [0, 30, 1000, 114424], 
                         labels=['Soluble', 'Moderate Soluble', 'Insoluble'])

#drop non-descriptor columns
data.drop(columns=['IUPAC_solid', 'SMILES_solid', 'IUPAC_anion','SMILES_anion', 
                   'IUPAC_cation', 'SMILES_cation', 'Mole_fraction','Weight_fraction', 
                   'Parts of Solvent', 'Classification'], axis=1, inplace=True)

#set data ID as index
data = data.set_index(data.ID.values)
data.drop(columns=['ID'], axis=1, inplace=True)

#new variable data_red - inspection of reduced solute+IL systems (without temperature)
data_red = data

#remove Temperature from ID

a=0
data_red_idx = []
for i in data_red.index:
    data_red_idx.append(re.sub(r'(\..*)|(K)', r'', data_red.index[a]))    
    a += 1

a=0
data_red_idx2 = []
for i in data_red.index:
    data_red_idx2.append(data_red_idx[a][:-4:])
    a += 1

#new ID    
data_red = data_red.set_index([data_red_idx2])    

#remove columns with zero values
zero_col = data_red.any()[(data_red.any()==False)].index
data_red = data_red.drop(zero_col, axis = 1)
data_red_col = data_red

#keep first data point for each system
data_dup = data_red.loc[~data_red.index.duplicated(keep='first')]

a=0
subst_count = []
subst = []
for i in data_dup.index:
    #unique systems
    subst.append(data_dup.index[a])
    #number of datapoints
    subst_count.append(len(data_red[data_red.index == data_dup.index[a]])) 
    a += 1

#remove Temperature physical descriptor
data_dup_mod = data_dup
data_dup_mod = data_dup_mod.drop(columns=['Temperature.K'])

#load dataset
def loadKS(input):
    try:
        X = np.loadtxt(input, delimiter="\t")
    except:
        X = pd.read_csv(input, header=None)
        return X.values
    return X

#computes Euclidean distance
def skdist(X, precomputed=False):
    if precomputed:
        return X
    return metrics.pairwise_distances(X, metric='euclidean', n_jobs=-1)

#Kennard Stone algorithm
def kenStone(X, k, precomputed=False):
    n = len(X) # number of samples
    print("Input Size:", n, "Desired Size:", k)
    assert n >= 2 and n >= k and k >= 2, "Error: number of rows must >= 2, k must >= 2 and k must > number of rows"
    # pair-wise distance matrix
    dist = skdist(X, precomputed)

    # get the first two samples
    i0, i1 = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
    selected = set([i0, i1])
    k -= 2
    # iterate find the rest
    minj = i0
    while k > 0 and len(selected) < n:
        mindist = 0.0
        for j in range(n):
            if j not in selected:
                mindistj = min([dist[j][i] for i in selected])
                if mindistj > mindist:
                    minj = j
                    mindist = mindistj
        print(selected, minj, [dist[minj][i] for i in selected])
        selected.add(minj)
        k -= 1
    print("selected samples indices: ", selected)
    # return selected samples
    if precomputed:
        return list(selected)
    else:
        return X[list(selected), :]

#write training set samples
def writeKS(output, X, precomputed=False):
    if precomputed:
        np.savetxt(output, X, fmt='%d')
    else:
        np.savetxt(output, X, fmt='%.5f')

#table system + count
duplicates_list = pd.DataFrame(
    {'system': subst,
     'point count': subst_count
    })

#compute Euclidean Distance
euc_dist = distance.squareform(distance.pdist(data_dup_mod, metric='euclidean'))
df_euc_dist = pd.DataFrame(euc_dist, index=duplicates_list['system'])

#change path to your directory
path2='/home/joao_ines/Documents/SLE/new_modeling/euc_dist_mod.csv'

#write distance matrix into a file
df_euc_dist.to_csv(path2, index=True, header=True)

#change path to your directory
path2 = '/home/joao_ines/Documents/SLE/new_modeling/euc_dist_mod.csv'

#load data set for training + test set selection with KS algorithm
X = loadKS(path2)

#delete first row and column from distance matrix
X = np.delete(X, 0, 0)
X = np.delete(X, 0, 1)

#compute KS-algorithm
Y = kenStone(X, round(179*0.8), precomputed=True)

#change path to your directory
path3='/home/joao_ines/Documents/SLE/new_modeling/kenStone_training_set_mod.txt'

#write indexes of training set systems
writeKS(path4, Y, precomputed=True)

number_systems = len(data_red_col.index.unique())

#change path to your directory
path4='Documents/SLE/new_modeling/kenStone_training_set_mod.txt'

# get training and test set indexes
tr_set_idx = pd.read_csv(path4,sep="/t", header=None).stack().tolist()
total_set_idx = list(range(0, data_dup.shape[0], 1))
test_set_idx = list(set(total_set_idx) - set(tr_set_idx))

#construct training and test set from KS algorithm indexes
training_data = data_red_col[data_red_col.index.isin(data_dup.iloc[tr_set_idx].index)]
test_data = data_red_col[data_red_col.index.isin(data_dup.iloc[test_set_idx].index)]

#change path to your directory
path5='/home/joao_ines/Documents/SLE/new_modeling/training_data_all_desc_mod.csv'
path6='/home/joao_ines/Documents/SLE/new_modeling/test_data_all_desc_mod.csv'

training_data.to_csv(path5, index=True, header=True)
test_data.to_csv(path6, index=True, header=True)
