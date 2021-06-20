#This is used to get the image names from the indices put in nbrs['s1']
#The image names are obtained from dict which got creare during feature_compute.py
import pickle
import json
from shutil import copy2
import os

with open('./nbrs_inc_s1.pkl','rb') as fp:
	nb = pickle.load(fp)

with open('./dict_index_exist_10k.json','r') as fp1:
	dict = json.load(fp1)

nbim={}
nbkeys = list(nb.keys())
print(len(nbkeys))
for key in range(len(nbkeys)):
	print(nbkeys[key])
	lval = nb[nbkeys[key]]
	print(lval)
	ims = []
	for li in lval:
		im = dict.keys()[list(dict.values()).index(li)]
		ims.append(im)
	nbim[nbkeys[key]]=ims
	
with open('nbrs_inc_s1_im.pkl','wb') as fp2:
	pickle.dump(nbim,fp2)