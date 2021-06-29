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

def optprob(exlist,Bfiles,fl,newPath,siftdic,lossdict,lossdict1,lossdict2,lossdict3,lossdict4,lossdict5,lossdict6):

	B=sorted(os.listdir(Bfiles))
	B=B[50:]
	threshold = 0.9
	rho = 0.50
	fraction = 0.2
	size_penalty = 0
	p=1
	wt = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 0.8, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8]
	for batch in range(0,len(B),200):


		lossesA=[]
		lossesB=[]
		
		B1=B[batch:batch+200]
		
		D1=np.zeros((len(B1),len(exlist)))
		D2=np.zeros((len(B1),len(B1)))
		D3=np.zeros((len(B1),len(exlist)))
		D4=np.zeros((len(B1),len(B1)))
	
		
		for f in range(0,len(exlist)):

			for f1 in range(0,len(B1)):
				imgs2=os.path.basename(B1[f1]).strip()+','+exlist[f].split('/')[1]+'_'+os.path.basename(exlist[f]).strip()
				
				if imgs2 in siftdic:
					#continue
					D1[f1][f]=siftdic[imgs2]
				else:
					print(imgs2)
					print("Error not found....")
					D1[f1][f]=sift_dissim(Bfiles+B1[f1],exlist[f])
					
			lossfile = exlist[f].split('/')[1]+'_'+os.path.basename(exlist[f]).strip()
			
			if lossfile not in lossdict: #Buffer of first 10 images
				lossesA.append(0.0)
				
			else:
				index = list(np.nonzero(lossdict6[lossfile])[0])
				
				lossA = (0.3*wt[index[0]]*float(lossdict[lossfile]))+(0.2*0.2*(float(lossdict1[lossfile])+float(lossdict2[lossfile])+float(lossdict3[lossfile])+float(lossdict4[lossfile])+float(lossdict5[lossfile])))
				lossesA.append(1.0/(1.0+math.exp(-lossA)))
			

		for f in range(0,len(B1)):
			for f1 in range(f,len(B1)):
				imgs1=os.path.basename(B1[f]).strip()+','+os.path.basename(B1[f1]).strip()
				if imgs1 in siftdic:
					#continue
					D2[f][f1]=siftdic[imgs1]
				else:	
					print("Error not found")
					D2[f][f1]=sift_dissim(Bfiles+B1[f],Bfiles+B1[f1])
				
			lossfile = Bfiles.split('/')[2]+'_'+os.path.basename(B1[f]).strip()
					
			if lossfile not in lossdict:#Buffer of first 10 images
				lossesB.append(0.0)
				
			else:
				index = list(np.nonzero(lossdict6[lossfile])[0])
				
				lossB = (0.3*wt[index[0]]*float(lossdict[lossfile]))+(0.2*0.2*(float(lossdict1[lossfile])+float(lossdict2[lossfile])+float(lossdict3[lossfile])+float(lossdict4[lossfile])+float(lossdict5[lossfile])))
				lossesB.append(1.0/(1.0+math.exp(-lossB)))
				
		#############################Forming output variables#############################################
		
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

		#Selecting the frame for which Z value is max for the incoming frame
		
		#print(N_p.value.shape[0])
		for i in range(N_p.value.shape[0]):
			#print(N_p.value[i])
			if round(N_p.value[i],1)>=0.9:
				copy2(Bfiles+B1[i],newPath)

		
def main():
	global siftdic
	direc = './example'
	cnt = 0
	folds = os.listdir(direc)

	with open('nbrs_set.pkl','rb') as fp:
		nb = pickle.load(fp)

	for fl in folds:

		exlist=[]
		if fl.startswith('episode'):

			siftdic = {}
			nbrs = nb[fl]
			for i in nbrs:
				imname = i.split('_image')[0]+'/CameraRGB/image'+ i.split('_image')[1]
				exlist.append('more_transformed/'+str(imname))
			
			exlist=exlist[:500]

			incPath=direc+'/'+fl+'/CameraRGB/'
			siftName=direc+'/siftcorr_incex_'+fl
			
			if os.path.exists(direc+'/'+fl+'/subset'):
				rmtree(direc+'/'+fl+'/subset')

			os.mkdir(direc+'/'+fl+'/subset')
			
			newPath = direc+'/'+fl+'/subset/'
		
			Bfiles=incPath
			
			if os.path.exists(siftName):
					with open(siftName, 'rb') as handle:
						siftdic = pickle.load(handle)
						#print(len(list(siftdic.keys())))
						
			if os.path.exists(direc+'/loss_subsetrel_json.json'):
					with open(direc+'/loss_subsetrel_json.json', 'r') as lossh:
		                                lossdict = json.load(lossh)
			if os.path.exists(direc+'/loss_subsetcen_json.json'):
		                        with open(direc+'/loss_subsetcen_json.json', 'r') as lossh:
		                                lossdict1 = json.load(lossh)
			if os.path.exists(direc+'/loss_subsetspd_json.json'):
		                        with open(direc+'/loss_subsetspd_json.json', 'r') as lossh:
		                                lossdict2 = json.load(lossh)
			if os.path.exists(direc+'/loss_subsethaz_json.json'):
		                        with open(direc+'/loss_subsethaz_json.json', 'r') as lossh:
		                                lossdict3 = json.load(lossh)
			if os.path.exists(direc+'/loss_subsetveh_json.json'):
		                        with open(direc+'/loss_subsetveh_json.json', 'r') as lossh:
		                                lossdict4 = json.load(lossh)
			if os.path.exists(direc+'/loss_subsetred_json.json'):
		                        with open(direc+'/loss_subsetred_json.json', 'r') as lossh:
		                                lossdict5 = json.load(lossh)
			if os.path.exists(direc+'/loss_subsetbucketrel_json.json'):
                                with open(direc+'/loss_subsetbucketrel_json.json', 'r') as lossh:
                                        lossdict6 = json.load(lossh)
				

			optprob(exlist,Bfiles,fl,newPath,siftdic,lossdict,lossdict1,lossdict2,lossdict3,lossdict4,lossdict5,lossdict6)
			cnt = cnt + 1
			
			print(str(cnt)+" done")


if __name__=='__main__':
	main()