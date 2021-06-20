import numpy as np
import faiss
import json
import time
import os
import pickle

m = 16                                   # number of subquantizers
n_bits = 8
d = 512
k = 1000

#Existing array
x_train = np.load('../feature_compute/res_train_exist_10k.npy')
print(x_train.shape)

x_train = np.float32(x_train)

#Incoming array as query
x_query = np.load('../feature_compute/res_train_set1_30k.npy')
nbrs = {}
#eps = 0
start = time.time()
x_query = np.float32(x_query)

print(x_query.shape)

pq = faiss.IndexPQ(d, m, n_bits)        # Create the index
pq.train(x_train)                       # Training
pq.add(x_train)                          # Populate the index
D, I = pq.search(x_query, k)
 #D and I are possibly number of queries x 1000 shape which means for each query image, find the closest 1000 (k) images
indices = I.flatten()
indices = list(set(list(indices))) #unique indices

print(len(indices))
	
Distn = {}
for ind in range(len(indices)):
	print(ind)
	img = indices[ind]
	dist = 0
	indcs = np.argwhere(I==img) #finding the indices in I where img is found
		
	for fi in range(len(indcs)):
		row = indcs[fi][0]
		col = indcs[fi][1]
		dist = dist + D[row][col] #adding up the distances
		
	Distn[indices[ind]] = dist
		

sorteddict = {k: v for k, v in sorted(Distn.items(), key=lambda item: item[1])}
	
nbrs['s1'] = list(sorteddict.keys())[:1000] #Top 1000 (can be less too) closest neighbours for the entire first set 

with open('nbrs_inc_s1.pkl','wb') as fp:
	pickle.dump(nbrs,fp,protocol=2)

#nbrs['s1'] has a list of indices
   
print(time.time()-start)