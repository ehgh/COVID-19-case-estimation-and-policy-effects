import numpy as np
import scipy
import scipy.io as sio
import sys
import pandas as pd
import pdb
import os
import datetime
import time

from tqdm import tqdm
from timeit import default_timer as timer 
from initialize import initialize
from Adaptive_SEIR import SEIR
from multiprocessing import Pool
from plotnine import *
from mizani.breaks import date_breaks
from mizani.formatters import date_format

np.set_printoptions(threshold = sys.maxsize, precision = 10, suppress=True)

#pdb.set_trace()

def estimator(num_ens=4000, 
              Td=6.,
              a=1.78,
              report_delay_bin=10, 
              sd_delay=0, 
              output_dir=time.strftime("%Y%m%d-%H%M%S"),
              sd_flag=True
              ):

    """
    estimation of daily cases using set parameter values

    Args:
        num_ens (int): number of ensembles in EnKF 
        Td (float): average reporting delay of documented cases
        a (float): shape parameter of gamma distribution of reporting delay, Td=a*b
        report_delay_bin (int): bin size of daily cases to add reporting delay (per state/ensemble) > 0
        sd_delay (int): number of days to shift social distancing 
        output_dir (str): directory to save output files, default is current date and time
        sd_flag (bool): flag to use social distancing in SEIR model           
    
    Returns:
        obs_temp (array): reported cases with delay
                          shape: (num_state, num_ens, num_times)
        daily_cases_u (array):  undocumented cases (no delay)
                                shape: (num_state, num_ens, num_times)
        x (array): state space vector of all ensembles, 
                   including S,E,Id,Iu,obs for each state and model parameters
                   shape: (num_ens, num_var)
                   num_var = num_state*5 + num_params
    """

    plot_ground_truth = False
    start = timer()

    #########################loading data and processing#########################

    #load mobility
    M_A = np.load('M_A.npy')
    M_G = np.load('M_G.npy')
    #load population
    pop = np.load('pop.npy')
    #load cumulative daily cases
    daily_cases = np.load('incidence.npy')
    #load social distancing measure
    sd = np.load('social_distancing.npy')
    #shift sd
    if sd_delay:
        sd = np.pad(sd, ((0,0),(sd_delay,0)), mode='constant')[:,:-int(sd_delay)]

    #sd_flag assertion
    if not sd_flag:
        sd = np.zeros(sd.shape)
    
    #########################initialize variables#########################
    
    #scale parameter of gamma distribution
    b = Td / a
    #generate gamma random numbers
    rnds = np.int64(np.ceil(np.random.gamma(a, b, (10000,))))
    #population of states
    pop0 = np.dot(pop, np.ones((1, num_ens)))
    #initialize state vector
    x, paramax, paramin = initialize(pop0, num_ens)
    #number of parameters (only model parameters)
    num_params = paramax.shape[0]
    #number of state variables (including S,E,Id,Iu,obs for each state and model parameters)
    num_var = x.shape[0]
    #number of states
    num_state = M_G.shape[0]
    #number of days
    num_times = sd.shape[1]
    obs_truth = daily_cases.copy().T
    #posterior parameters
    param_post = np.zeros((num_var, num_ens, num_times))
    pop_post = np.zeros((num_state, num_ens, num_times))
    
    #########################start estimation iteration#########################
    
    #first guess of state space
    x,_,_ =initialize(pop0, num_ens)
    np.save('initial_state_vector.npy', x)
    #Begin looping through observations
    pop = pop0
    #records of reported cases with delay
    obs_temp = np.zeros((num_state, num_ens, num_times))
    #records of undocumented cases (undocumented don't have delays)
    daily_cases_u = np.zeros((num_state, num_ens, num_times))

    for t in tqdm(np.arange(0, num_times)):
    
        x, pop, obs_cnt, daily_cases_u[:,:,t] = SEIR(x, M_G, M_A, pop, t, pop0, sd=sd)
        #add reporting delay
        X,Y = obs_cnt.nonzero()
        values = (obs_cnt[X,Y]).astype(int)
        weights_mul = np.ones_like(values) * report_delay_bin
        weights_res = np.ones_like(values)
        values_mul = (values/report_delay_bin).astype(int)
        values_res = (values%report_delay_bin)
        [X_mul,Y_mul] = np.repeat(np.vstack([X,Y]), values_mul, axis=1)
        [X_res,Y_res] = np.repeat(np.vstack([X,Y]), values_res, axis=1)
        weights_mul = np.repeat(weights_mul, values_mul)
        weights_res = np.repeat(weights_res, values_res)
        X = np.hstack((X_mul, X_res))
        Y = np.hstack((Y_mul, Y_res))
        weights = np.hstack((weights_mul, weights_res))
        Z = np.random.choice(rnds, size=X.shape[0]) + t
        mask = Z<num_times
        flat = np.ravel_multi_index((X[mask], Y[mask], Z[mask]), obs_cnt.shape+(num_times,))
        obs_temp = obs_temp + np.bincount(flat,weights[mask],obs_cnt.size*num_times).reshape(*obs_cnt.shape,num_times)
                                
        #correct lower/upper bounds of the unrealistic parameter values
        x = checkbound(x, pop) 
        param_post[:, :, t] = x
        pop_post[:, :, t] = pop
    
    return obs_temp, daily_cases_u, param_post, pop_post

def checkbound(x, pop):

    #S,E,Id,Iu,obs,...,beta,mu,theta,Z,alpha,D
    num_state = pop.shape[0]

    #S, E, Id, Iu, and obs
    x_slice = x[:num_state*5, :]
    x_slice[x_slice<0] = 0
    #S
    x_S = x[:num_state*5:5, :]
    x_S[x_S>pop] = pop[x_S>pop]
        
    return x