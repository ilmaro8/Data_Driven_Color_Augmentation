import pandas as pd
import numpy as np
import os
from PIL import Image
import albumentations as A
from scipy.spatial import KDTree, cKDTree
import pickle
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import argparse
import sys, getopt

argv = sys.argv[1:]


#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-i', '--INPUT', help='input to extend',type=str, default='input.pickle')
parser.add_argument('-o', '--OUTPUT', help='where to store file',type=str, default='output.pickle')
parser.add_argument('-d', '--DATA_TO_ADD', help='csv_file',type=int, default=32)


args = parser.parse_args()

INPUT_DATA = args.DATA_TO_ADD
OUTPUT_FILE = args.OUTPUT
INPUT_FILE = args.INPUT

def H_E_Staining(img, Io=240, alpha=1, beta=0.15):

	# define height and width of image
	h, w, c = img.shape

	# reshape image
	img = img.reshape((-1,3))

	# calculate optical density
	OD = -np.log((img.astype(np.float)+1)/Io)

	# remove transparent pixels
	ODhat = OD[~np.any(OD<beta, axis=1)]

	# compute eigenvectors
	eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

	#eigvecs *= -1

	#project on the plane spanned by the eigenvectors corresponding to the two 
	# largest eigenvalues    
	That = ODhat.dot(eigvecs[:,1:3])

	phi = np.arctan2(That[:,1],That[:,0])

	minPhi = np.percentile(phi, alpha)
	maxPhi = np.percentile(phi, 100-alpha)

	vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
	vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

	# a heuristic to make the vector corresponding to hematoxylin first and the 
	# one corresponding to eosin second
	if vMin[0] > vMax[0]:
		HE = np.array((vMin[:,0], vMax[:,0])).T
	else:
		HE = np.array((vMax[:,0], vMin[:,0])).T

	return HE

fname = INPUT_FILE

with open(fname, 'rb') as f:
	kdtree = pickle.load(f)

HEs_general = kdtree.data

input_csv = INPUT_DATA
input_data = pd.read_csv(input_csv, sep=',',header=None).values

def extend_stains(kdtree, new_data, save_new_array = False, PERC=1.0):
	
	HEs_new_stains = []
	
	threshold_value = int(len(new_data)*PERC)
	
	i = 0
		
	HEs_general = kdtree.data
	
	np.random.shuffle(new_data)
	
	for i in tqdm(range(threshold_value)):

		patch = new_data[i,0]
		
		img = Image.open(patch)
		img_np = np.asarray(img)
		
		HE = H_E_Staining(img_np)
		
		HE = np.reshape(HE, 6)
		HEs_new_stains.append(HE)
		
		img.close()
		
		#i = i + 1

	HEs_new_stains = np.array(HEs_new_stains)
	HEs_general = np.append(HEs_general,HEs_new_stains,axis=0)
	
	new_kdtree = cKDTree(HEs_general)
	
	if (save_new_array==True):
		
		fname = OUTPUT_FILE
		with open(fname, 'wb') as f:
			pickle.dump(kdtree, f)
		
	return new_kdtree

start_time = time.time()
new_kdtree = extend_stains(kdtree, input_data, save_new_array = True, PERC=1.0)
elapsed_time = time.time() - start_time
print("elapsed time " + str(elapsed_time))