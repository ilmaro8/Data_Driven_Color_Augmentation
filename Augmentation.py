import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import warnings 

from scipy.spatial import KDTree, cKDTree

warnings.filterwarnings("ignore")

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

def unique_elements(array):
	
	_, counts = np.unique(array, return_counts=True)
	
	b = True
	
	for c in counts:
		
		if (c>1):
			
			b = False
	
	return b

def normalizeStaining(img, HERef, Io=240, alpha=1, beta=0.15):

	maxCRef = np.array([1.9705, 1.0308])
	
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
	
	# rows correspond to channels (RGB), columns to OD values
	Y = np.reshape(OD, (-1, 3)).T
	
	# determine concentrations of the individual stains
	C = np.linalg.lstsq(HE,Y, rcond=None)[0]
	
	# normalize stain concentrations
	maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
	tmp = np.divide(maxC,maxCRef)
	C2 = np.divide(C,tmp[:, np.newaxis])
	
	# recreate the image using reference mixing matrix
	Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
	Inorm[Inorm>255] = 254
	Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  

	return Inorm

def new_color_augmentation(patch_np, kdtree, alpha, beta, shift_value=70, threshold=1000):
	
	b = False
	i = 0
	
	pipeline_transform_ = A.Compose([
		A.HueSaturationValue(hue_shift_limit=(-shift_value,shift_value),sat_shift_limit=(-shift_value,shift_value),val_shift_limit=(-shift_value,shift_value),always_apply=True),
	])
	
	while (b==False and i<threshold):
		
		A_np = pipeline_transform_(image=patch_np)['image']

		try:
			HE_ref = H_E_Staining(A_np)
			point = np.reshape(HE_ref, 6)
			
			_, points_indeces = kdtree.query(point, k = alpha, distance_upper_bound = beta)
			
			if (unique_elements(points_indeces)):
				
				b = True
				
			else:
				
				i = i + 1

		except:
			print("H&E not valid")
	

	return A_np, HE_ref


def new_stain_augmentation(patch_np, kdtree, alpha, beta, sigma1, sigma2, threshold=1000000):
	
	b = False
	i = 0

	
	data = kdtree.data
	
	while (b==False and i<threshold):
		
		idx_HE_ref = np.random.choice(data.shape[0])
		
		HE_ref = data[idx_HE_ref]
		#print(HE_ref)
		HE_ref = np.reshape(HE_ref, (3,2))
		
		alpha_sig = np.random.uniform(1 - sigma1, 1 + sigma1)
		beta_sig = np.random.uniform(-sigma2, sigma2)
		
		HE_ref *= alpha_sig 
		HE_ref += beta_sig
		
		#print(HE_ref)
		point = np.reshape(HE_ref, 6)
		
		_, points_indeces = kdtree.query(point, k = alpha, distance_upper_bound = beta)
		
		if (unique_elements(points_indeces)):
			
			b = True
			
		else:
			
			i = i + 1
	
	A_np = normalizeStaining(patch_np, HE_ref)

	return A_np, HE_ref

