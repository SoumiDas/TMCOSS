from shutil import copy2
import os
import csv
import numpy as np
import cv2
#import cvxpy as cvx
import numpy as np
import pandas as pd
from skimage.measure import compare_ssim
from skimage.transform import resize
from scipy.stats import wasserstein_distance
from scipy.misc import imsave
from scipy.ndimage import imread
#import test2
import time
import json
import pickle
import sys

from concurrent.futures import ThreadPoolExecutor
import threading
from threading import Thread

countLock = threading.Lock()
count=0

#executor = ThreadPoolExecutor(max_workers=64)

siftdic={}
structdic={}

def old_new(tup):

			try:
				A,B1,f=tup
				global count
				a=0
				global siftdic
				
				
				skip=f
				start=time.time()
				print('Reading file : ',f)
				for f1 in range(0,len(B1)):

					imgs1=os.path.basename(A[f]).strip()+','+os.path.basename(B1[f1]).strip()
					imgs2=os.path.basename(B1[f1]).strip()+','+os.path.basename(A[f]).strip()
					if imgs2 in siftdic:
						a=1
					else:
						siftstart=time.time()
						#siftdic[imgs1]=sift_dissim(Afiles+A[f],Bfiles+B1[f1])
						siftdic[imgs2]=sift_dissim(B1[f1],A[f])
				
				
			except Exception as e:
				with open("log_old_new.txt",'a') as w:
					w.write(str(e))
		
			

def new_new_sift(tup):
	try:
			B1,f=tup
			#global siftdic,structdic,count
			global count

			print('Reading file : ',B1[f])
			
			for f1 in range(0,len(B1)):
				imgs1=os.path.basename(B1[f]).strip()+','+os.path.basename(B1[f1]).strip()
				print(imgs1)
				#print(B1[f])
				if imgs1 in siftdic:
					#print("Present")
					continue
				else:	
					siftdic[imgs1]=sift_dissim(B1[f],B1[f1])
			
	except Exception as e:
				with open("log_new_new_sift.txt",'a') as w:
					w.write(str(e))				

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



def optprob(exlist,B,executor,siftdic):

	
	for batch in range(0,len(B),1000):

		B1=B[batch:batch+1000]
			
		for f in range(0,len(exlist)):
			tup=(exlist,B1,f)
			ts=executor.submit(old_new,tup)
			#exit(1)
			#print ("ts",ts)
						
		for f in range(0,len(B1)):
			#print("Inside")
			tup=(B1,f)
			tnnsi=executor.submit(new_new_sift,tup)
			#print("tonsi ",tnnsi)
	
def main():
	global siftdic
	
	with open('./nbrs_inc_s1_im.pkl','rb') as fp:
		nb = pickle.load(fp)

	nbrs = nb['s1']

	exlist = []
	inlist = []

	for i in nbrs:
		exlist.append('./exs/'+i.strip())
	#print("Loaded")
	exlist = sorted(exlist)
	exlist = exlist[:500] #Existing Set - take a max of 500, append paths
	with open('./set1.txt','r') as fp1:
		for line in fp1:
			inlist.append('../../../'+line.strip())

	inlist = sorted(inlist)
	#for group in range(0,len(inlist),1000):
	for group in range(0,4000,1000): #Incoming set compute in batches
		inlistg = inlist[group:group+1000]

		siftName='siftval_s1_'+str(group)

		if os.path.exists(siftName):
			with open(siftName, 'rb') as handle:
				siftdic = pickle.load(handle)
		else:
			siftdic = {}
		
		start=time.time()
		executor = ThreadPoolExecutor(max_workers=64)
		
		optprob(exlist,inlistg,executor, siftdic)
		executor.shutdown(True)
			
		with open(siftName, 'wb') as handle:
				pickle.dump(siftdic, handle, protocol=pickle.HIGHEST_PROTOCOL)
		print(time.time()-start)

if __name__=='__main__':
	main()