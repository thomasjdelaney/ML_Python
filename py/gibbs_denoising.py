"""
A script for adding noise to an image, then using Gibbs sampling to denoise it.
	execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
"""
import os
execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave

root_dir = os.path.join(os.environ['HOME'], 'ML_Python')
image_dir = os.path.join(root_dir, 'images')

def addGaussianNoise(im, prop, varSigma):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    e = varSigma*np.random.randn(np.prod(im.shape)).reshape(im.shape)
    im2 = np.copy(im).astype('float')
    im2[index] += e[index]
    return im2

def addSaltNPepperNoise(im, prop):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    im2 = np.copy(im)
    im2[index] = 1-im2[index]
    return im2

def getNeighbours(i,j,M,N,size=4):
    if size==4:
        if (i==0 and j==0):
            n=[(0,1), (1,0)]
        elif i==0 and j==N-1:
            n=[(0,N-2), (1,N-1)]
        elif i==M-1 and j==0:
            n=[(M-1,1), (M-2,0)]
        elif i==M-1 and j==N-1:
            n=[(M-1,N-2), (M-2,N-1)]
        elif i==0:
            n=[(0,j-1), (0,j+1), (1,j)]
        elif i==M-1:
            n=[(M-1,j-1), (M-1,j+1), (M-2,j)]
        elif j==0:
            n=[(i-1,0), (i+1,0), (i,1)]
        elif j==N-1:
            n=[(i-1,N-1), (i+1,N-1), (i,N-2)]
        else:
            n=[(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
        return n
    if size==8:
        print('Not yet implemented\n')
    	return -1

def loadNoisyImage(file_name, dir=image_dir):
	return imread(os.path.join(dir, file_name))/255.0

original_image = imread(os.path.join(image_dir, 'dachshund_bw.jpg'))/255
gaussian_image = addGaussianNoise(original_image, 0.7, 0.1)
noisy_image = addSaltNPepperNoise(original_image, 0.7)
gaussian_path = os.path.join(image_dir, 'dachshund_gaussian.jpg')
noisy_path = os.path.join(image_dir, 'dachshund_noisy.jpg')
imsave(gaussian_path, gaussian_image*255)
imsave(noisy_path, noisy_image*255)
h = 0
beta = 1.0
eta = 2.1
M,N = noisy_image.shape
bias_plus = 0
local_corr_plus = 0
image_corr_plus = 0
for m in range(0,M):
	for n in range(0,N):
		x_i = 1
		neighbours = getNeighbours(m,n,M,N)
		bias_plus += x_i
		for (n_m, n_n) in neighbours:
			neighbour_value = noisy_image[n_m, n_n]
			local_corr_plus += x_i*neighbour_value
		pixel_value = noisy_image[m,n]
		image_corr_plus += x_i*pixel_value
