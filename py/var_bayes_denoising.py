"""
A script for adding noise to an image, then using variational Bayes to denoise it.
	execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
"""
import os, argparse
execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
from scipy.special import expit

root_dir = os.path.join(os.environ['HOME'], 'ML_Python')
image_dir = os.path.join(root_dir, 'images')

parser = argparse.ArgumentParser(description="For adding noise to an image, then denoising it using Gibbs sampling.")
parser.add_argument('-f', '--picture_file', help='The picture to which to add noise.', type=str, default='dachshund_bw.jpg')
parser.add_argument('-p', '--proportion', help='The proportion of bits to add noise.', type=float, default=0.7)
parser.add_argument('-v', '--noise_variance', help='Variance for Gaussian noise.', type=float, default=0.1)
parser.add_argument('-t', '--num_iterations', help='The number of iterations to use.', type=int, default=10)
parser.add_argument('-d', '--debug', help='Enter debug mode.', action='store_true', default=False)
args = parser.parse_args()

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

def getNoisyImages(picture_file, image_dir, prop, var, is_save=False):
	original_image = imread(os.path.join(image_dir, picture_file))/255
	gaussian_image = addGaussianNoise(original_image, prop, var)
	noisy_image = addSaltNPepperNoise(original_image, prop)
	if is_save:
		gaussian_path = os.path.join(image_dir, 'dachshund_gaussian.jpg')
		noisy_path = os.path.join(image_dir, 'dachshund_noisy.jpg')
		imsave(gaussian_path, gaussian_image*255)
		imsave(noisy_path, noisy_image*255)
	return original_image, gaussian_image, noisy_image

def getRescaledImages(gaussian_image, noisy_image):
	rescaled_gaussian = np.interp(gaussian_image, (0, 1), (-1, 1))
	rescaled_noisy = np.interp(noisy_image, (0, 1), (-1, 1))
	return rescaled_gaussian, rescaled_noisy

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
        return np.array(n)
    if size==8:
        print('Not yet implemented\n')
    	return -1

def loadNoisyImage(file_name, dir=image_dir):
	return imread(os.path.join(dir, file_name))/255.0

def plotNoisyAndDenoised(num_passes, noisy_image, gaussian_image, denoised_noisy, denoised_gaussian):
	ax = fig.add_subplot(221)
	plt.title('Num Passes = ' + str(num_passes))
	ax.imshow(noisy_image, cmap='gray')
	ax = fig.add_subplot(222)
	ax.imshow(gaussian_image, cmap='gray')
	ax = fig.add_subplot(223)
	ax.imshow(denoised_noisy, cmap='gray')
	ax = fig.add_subplot(224)
	ax.imshow(denoised_gaussian, cmap='gray')

def getPermutedIndices(image_shape):
	return np.array(np.unravel_index(np.random.permutation(np.prod(image_shape)), image_shape)).T

def getComputeParam(m, n, M, N, mus):
	neighbours = getNeighbours(m, n, M, N)
	return mus[neighbours[:,0], neighbours[:,1]].sum()

def imageCorrelation(x_i, y_i): return x_i * y_i

def corrDifference(pixel_value):
	return 0.5*(imageCorrelation(1, pixel_value) - imageCorrelation(-1, pixel_value))

def getVariationalParam(compute_param, pixel_value):
	return np.tanh(compute_param + corrDifference(pixel_value))

def main():
	original_image, gaussian_image, noisy_image = getNoisyImages(args.picture_file, image_dir, args.proportion, args.noise_variance)
	rescaled_gaussian, rescaled_noisy = getRescaledImages(gaussian_image, noisy_image)
	M, N = noisy_image.shape
	mus = np.random.uniform(-1, 1, size=(M, N))
	denoised_noisy = rescaled_noisy
	denoised_gaussian = rescaled_gaussian
	fig = plt.figure()
	for t in range(0,args.num_iterations):
		for m,n in it.product(range(M), range(N)):
			gaussian_pixel_value = denoised_gaussian[m, n]
			compute_param = getComputeParam(m, n, M, N, mus)
			variational_param = getVariationalParam(compute_param, gaussian_pixel_value)
			mus[m, n] = variational_param
			denoised_gaussian[m, n] = expit(2*(getComputeParam(m, n, M, N, mus) + corrDifference(gaussian_pixel_value)))
		plotNoisyAndDenoised(t+1, noisy_image, gaussian_image, denoised_noisy, denoised_gaussian)
		plt.pause(0.05)

if not(args.debug):
	main()
