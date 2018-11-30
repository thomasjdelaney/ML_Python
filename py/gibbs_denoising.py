"""
A script for adding noise to an image, then using Gibbs sampling to denoise it.
"""
import os
execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
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

def saveNoisyImages(image, prop, var_sigma, dir=image_dir):
    pepe_gaussian = addGaussianNoise(image, prop, var_sigma)*255
    pepe_gaussian_path = os.path.join(dir, 'dachshund_gaussian.jpg')
    imsave(pepe_gaussian_path, pepe_gaussian)
    pepe_noisy = addSaltNPepperNoise(image, prop)*255
    pepe_noisy_path = os.path.join(dir, 'dachshund_noisy.jpg')
    imsave(pepe_noisy_path, pepe_noisy)
    return pepe_gaussian_path, pepe_noisy_path

def neighbours(i,j,M,N,size=4):
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

original_image = imread(os.path.join(image_dir, 'dachshund_bw.jpg'))/255
saveNoisyImages(original_image, 0.7, 0.1)
