#Before running this code, find loss on the entire training set (easier than finding which images to find loss for)
#Ideally, you should be having the losses for the current existing set and incoming set's images
#The first model will be the one you train on Existing set, using which loss has to be found
#Following this, you need to train on the existing set + subset you get in the earlier run and use that model to find loss for the current subset computation

from shutil import copy2
from shutil import copytree
from shutil import rmtree
import os
import csv
import numpy as np
import cv2
import cvxpy as cvx
import numpy as np
import pandas as pd
from skimage.measure import compare_ssim
from skimage.transform import resize


#import test2
import time
import json
import pickle
import sys
import math

def sift_dissim(path_a, path_b):
  '''
  Use SIFT features to measure image similarity
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    TODO
  '''
  # initialize the sift feature detector
  orb = cv2.ORB_create()

  # get the images
  img_a = cv2.imread(path_a)
  img_b = cv2.imread(path_b)

  # find the keypoints and descriptors with SIFT
  kp_a, desc_a = orb.detectAndCompute(img_a, None)
  kp_b, desc_b = orb.detectAndCompute(img_b, None)

  # initialize the bruteforce matcher
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  # match.distance is a float between {0:100} - lower means more similar
  matches = bf.match(desc_a, desc_b)
  similar_regions = [i for i in matches if i.distance < 50]
  if len(matches) == 0:
    return 0
  return 1-(float(len(similar_regions)) / float(len(matches)))

def optprob(exlist,B,newPath,siftdic,lossdict):

	threshold = 0.9
	rho = 0.50
	#lbda=0.1
	fraction = 0.2  # fraction of original points to be selected
	size_penalty = 0
	p=1

	for batch in range(0,len(B),200):


		lossesA=[]
		lossesB=[]
		
		B1=B[batch:batch+200]
		
		D1=np.zeros((len(B1),len(exlist)))
		D2=np.zeros((len(B1),len(B1)))
		
		for f in range(0,len(exlist)):

			for f1 in range(0,len(B1)):
				#imgs1=os.path.basename(A[f]).strip()+','+os.path.basename(B1[f1]).strip()
				imgs2=os.path.basename(B1[f1]).strip()+','+os.path.basename(exlist[f]).strip()
				
				if imgs2 in siftdic:
					#continue
					D1[f1][f]=siftdic[imgs2]
				else:
					print(imgs2)
					print("Error not found....")
					D1[f1][f]=sift_dissim(B1[f1],exlist[f])
					
			lossfile = os.path.basename(exlist[f]).strip()
			#print(lossfile)
			if lossfile not in lossdict:
				print("Not in dict")
				print(lossfile)
				lossesA.append(0.0)
			else:
				lossesA.append(1.0/(1.0+math.exp(-float(lossdict[lossfile]))))				

		for f in range(0,len(B1)):

			for f1 in range(f,len(B1)):
				imgs1=os.path.basename(B1[f]).strip()+','+os.path.basename(B1[f1]).strip()
				if imgs1 in siftdic:
					#continue
					D2[f][f1]=siftdic[imgs1]
				else:
					print(imgs1)	
					print("Error not found")
					D2[f][f1]=sift_dissim(Bfiles+B1[f],Bfiles+B1[f1])
				
			lossfile = os.path.basename(B1[f]).strip()
					
			if lossfile not in lossdict:
				print("Not in dict")
				print(lossfile)
				lossesB.append(0.0)
			
			else:
				lossesB.append(1.0/(1.0+math.exp(-float(lossdict[lossfile]))))
		#############################Forming output variables#############################################
		start=time.time()
		print("Defining variables to optimise")
	
		Zo=cvx.Variable((len(B1),len(exlist)))
		Zn=cvx.Variable((len(B1),len(B1)))

		No=cvx.sum(Zo, axis=0)
		Nn=cvx.sum(Zn, axis=0) 

		Sn=(1 / threshold)*cvx.minimum(threshold,Nn)
		So=(1 / threshold)*cvx.minimum(threshold,No)	

		#############################Forming constraints#############################################

		print("Defining constraints")

		M = cvx.sum(Zn, axis=1) + cvx.sum(Zo,axis=1)

		#############################Forming objective function###########################################


		#print("Constraints:",constraints)
		print("Optimising objective function")


		obj1 = rho * ( (1 / (len(exlist) * len(B1))) * cvx.sum(cvx.multiply(D1, Zo))  +  (1 / (len(B1) * len(B1))) * cvx.sum(cvx.multiply(D2, Zn)))
		obj2 = (1 - rho) * ( (1 / len(B1)) * cvx.sum(cvx.multiply(Sn, lossesB)) + (1/len(exlist)) * cvx.sum(cvx.multiply(So,lossesA)))

		N_p = cvx.norm(Zn, p=1, axis=0)  # sum over i, estimate of j's representativeness
		#est_size = cp.mixed_norm(Z.T, 2, 1)
		est_size = cvx.norm(N_p, p=1)
		obj3 = size_penalty * est_size

		obj = obj1 - obj2 + obj3

		constraints = [0.0<=Zn, Zn<=1.0, 0.0<= Zo, Zo<=1.0, M==1, est_size <= fraction * len(B1)]
		#constraints = [0.0<=Zn, Zn<=1.0, 0.0<= Zo, Zo<=1.0, est_size <= fraction * len(B1)]
		objective = cvx.Minimize(obj)
		
		prob = cvx.Problem(objective, constraints)
		try:
			prob.solve()  # Returns the optimal value.
		except:
			prob.solve(solver=cvx.SCS)

		print("status:", prob.status)
		print("optimal value", prob.value)

		print('Optimisation in time:',time.time()-start)
	
		for i in range(N_p.value.shape[0]):
			#print(N_p.value[i])
			if round(N_p.value[i],1)>=0.9:
				copy2(B1[i],newPath) #copying selected images to subset folder denoted by newPath

		
def main():
	global siftdic
	#direc = '../more_transformed/'
	cnt = 0

	exlist = []

	with open('./nbrs_inc_s1_im.pkl','rb') as fp:
		nb = pickle.load(fp)

	nbrs = nb['s1']

	exlist = []
	inlist = []

	for i in nbrs:
		exlist.append('../../../../bddreduce100k/images/train/'+i.strip())

	exlist = sorted(exlist)
	exlist = exlist[:500]

	#os.mkdir('../../../../bddsub100k/images/tcl/train/')

	newPath = '../../../../bddsub100k/images/tcl/train/'

	with open('./set1.txt','r') as fp1:
		for line in fp1:
			inlist.append('../../../../'+line.strip())

	inlist = sorted(inlist)

	for group in range(0,len(inlist),1000):
		inlistg = inlist[group:group+1000]

		siftName='../../sift_compute/siftval_s1_'+str(group)

		if os.path.exists(siftName):
			with open(siftName, 'rb') as handle:
				siftdic = pickle.load(handle)

		if os.path.exists('loss_subset_json.json'): #dictionary with key as file name, value is the loss value
			with open('loss_subset_json.json', 'r') as lossh:
				lossdict = json.load(lossh)

		optprob(exlist,inlistg,newPath,siftdic,lossdict)
		cnt = cnt + 1

	print(str(cnt)+" done")

if __name__=='__main__':
	main()