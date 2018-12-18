"""
A script for comparing poisson and gaussian distributions to understand the kl-divergence measure
    execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
"""
import os
execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
import argparse
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import entropy, norm, poisson

parser = argparse.ArgumentParser(description='Script for comparing Gaussian and Poisson distributions using the k-l divergence.')
parser.add_argument('-m', '--mean_range', help='The range of the means to test out.', type=int, nargs=2, default=[1,20])
parser.add_argument('-d', '--debug', help='Enter debug mode.', action='store_true', default=False)
args = parser.parse_args()

def getXAxesFromMean(distn_mean):
    std = np.sqrt(distn_mean)
    interval_start, interval_end = distn_mean + np.array([-4*std, 4*std]) # include mass within 3 standard deviations.
    interval_start = np.floor(interval_start).astype(int)
    interval_end = np.ceil(interval_end).astype(int)
    x_axis_points = np.linspace(interval_start, interval_end, 1000)
    x_axis_discrete = np.array(range(interval_start, interval_end+1))
    x_axis_discrete_gaussian = np.array(range(interval_start, interval_end+2))-0.5
    return x_axis_points, x_axis_discrete, x_axis_discrete_gaussian

def KLDivergence(p_distn, q_distn):
    symmetric_kl = entropy(p_distn, q_distn) + entropy(q_distn, p_distn)
    standard_kl = entropy(p_distn, q_distn)
    return symmetric_kl, standard_kl

def plotProbabilities(x_axis_discrete, poisson_probs, gaussian_probs):
    plt.step(x_axis_discrete, poisson_probs, color='blue', where='mid', label='Poisson probabilities')
    plt.step(x_axis_discrete, gaussian_probs, color='orange', where='mid', label='Gaussian probabilities')
    plt.xlabel('$x$'); plt.ylabel('$P(x)$')
    plt.legend()

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    fig = plt.figure(); plt.tight_layout();
    distns_ax = plt.subplot(211); divergence_ax = plt.subplot(212)
    means_to_test = range(args.mean_range[0], args.mean_range[1])
    num_means = len(means_to_test)
    divergences = np.zeros(num_means)
    for i in range(num_means):
        mean_to_test = means_to_test[i]
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'constructing distributions with mean = ' + str(mean_to_test) + ' ...')
        poisson_distn = poisson(mean_to_test)
        gaussian_distn = norm(mean_to_test, np.sqrt(mean_to_test))
        x_axis_points, x_axis_discrete, x_axis_discrete_gaussian = getXAxesFromMean(mean_to_test)
        poisson_probs = poisson_distn.pmf(x_axis_discrete)
        gaussian_probs = np.diff(gaussian_distn.cdf(x_axis_discrete_gaussian))
        symmetric_kl, standard_kl = KLDivergence(poisson_probs, gaussian_probs)
        divergences[i] = standard_kl
        plt.subplot(distns_ax)
        plt.cla()
        plotProbabilities(x_axis_discrete, poisson_probs, gaussian_probs)
        plt.pause(0.05)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'plotting divergence across means...')
    plt.subplot(divergence_ax)
    plt.plot(means_to_test, divergences, color='red')
    plt.xlabel('$\mu$'); plt.ylabel('$D_{KL}(Poiss(\mu), \mathcal{N}(\mu, \mu))$')
    plt.show(block=False)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Final divergence = ' + str(divergences[-1]))

if not(args.debug):
    main()
