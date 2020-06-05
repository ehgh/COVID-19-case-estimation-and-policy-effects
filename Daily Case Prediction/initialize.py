import numpy as np
import pandas as pd
import scipy
import scipy.io as sio
import random


def initialize(pop, num_ens):

    #Initialize the Adaptive SEIR model
    M_G = np.load('M_G.npy')
    M_A = np.load('M_A.npy')
    #load mobility
    num_loc = pop.shape[0]
    # S,E,Id,Iu,obs,...,beta,mu,theta,Z,alpha,D
    # prior range
    Slow = 1.0
    Sup = 1.0
    #susceptible fraction
    Elow = 0.
    Eup = 0.
    #exposed
    Irlow = 0.
    Irup = 0.
    #documented infection
    Iulow = 0.
    Iuup = 0.
    #undocumented infection
    obslow = 0.
    obsup = 0.
    #reported case
    paramin, paramax = param_range()
    #range of model state including variables and parameters
    xmin = np.empty((0, 1))
    xmax = np.empty((0, 1))
    for i in np.arange(0, num_loc):
        xmin = np.vstack((xmin, [[Slow*pop[i, 0]]], [[Elow*pop[i, 0]]], [[Irlow*pop[i, 0]]], [[Iulow*pop[i, 0]]], [[obslow]]))
        xmax = np.vstack((xmax, [[Sup*pop[i, 0]]], [[Eup*pop[i, 0]]], [[Irup*pop[i, 0]]], [[Iuup*pop[i, 0]]], [[obsup]]))
        
    xmin = np.vstack((xmin, paramin))
    xmax = np.vstack((xmax, paramax))
    
    Id_seed = np.around(np.load('initial_Id.npy'))
    Iu_seed = np.around(np.load('initial_Iu.npy'))
    seedid = np.around(np.load('seedid.npy'))
    Id_seed = Id_seed[seedid].reshape(-1, 1)
    Iu_seed = Iu_seed[seedid].reshape(-1, 1) 

    #E
    xmin[(seedid)*5+1] = 0.
    xmax[(seedid)*5+1] = Iu_seed
    #Id
    xmin[(seedid)*5+2] = 0.
    xmax[(seedid)*5+2] = 0.
    #Iu
    xmin[(seedid)*5+3] = 0.
    xmax[(seedid)*5+3] = Iu_seed
    #Latin Hypercubic Sampling
    x = lhsu(xmin, xmax, num_ens)
    x = x.conj().T
    for i in np.arange(0, num_loc):
        x[i*5:i*5+4, :] = np.round(x[i*5:i*5+4, :])
    
    #seeding in other cities
    C = M_G[:, seedid, 1] + M_A[:, seedid, 1]
    #day 0
    for i in np.arange(0, num_loc):
        if not i in seedid:
            #E
            Eseed = x[(seedid)*5+1, :]
            x[i*5+1, :] = np.round(np.dot(C[i,:] * 3, (Eseed / pop[seedid])))
            #Iu
            Iuseed = x[(seedid)*5+3, :]
            x[i*5+3, :] = np.round(np.dot(C[i,:] * 3, (Iuseed / pop[seedid])))
    #set Id seed for first occurrence states: Baden[0]: 1, Bavaria[1]: 14, North Rhine[9]: 1
    x[seedid*5+2, :] = Id_seed

    return x, paramax, paramin

def lhsu(xmin, xmax, nsample):

    nvar = len(xmin)
    ran = np.random.rand(nsample, nvar)
    s = np.zeros((nsample, nvar))
    for j in np.arange(0, nvar):
        idx = np.array(random.sample(range(1, nsample+1), nsample))
        P = (idx.conj().T - ran[:, j]) / nsample
        s[:,j] = xmin[j] + P * (xmax[j] - xmin[j])
    
    return s

def param_range():

    #This section can be used to set inference prior ranges
    #transmission rate
    betalow = 0.8
    betaup = 1.2
    #relative transmissibility
    mulow =  0.2
    muup = 1.
    #movement factor ground
    thetaglow = 1.
    thetagup = 1.8
    #movement factor air
    thetaflow = 1.
    thetafup = 1.8
    #latency period
    Zlow = 2.
    Zup = 5.
    #social distancing effect
    gammalow =  0.6
    gammaup = 1.
    #infectious period
    Dlow = 2.
    Dup = 5.
    #alpha_j
    alpha_jlow = [[0.5]] * 16
    alpha_jup = [[1.]] * 16
    
    #model parameters prior range for inference
    paramin = np.array([[betalow], [mulow], [thetaglow], [thetaflow], [Zlow], [gammalow], [Dlow]] + alpha_jlow)
    paramax = np.array([[betaup], [muup], [thetagup], [thetafup], [Zup], [gammaup], [Dup]] + alpha_jup)
    
    #model parameters inferred and usef for estimation
    paramin = np.array([[0.95],[0.22],[1.08],[1.63],[2.41],[0.89],[2.42],[0.71],[0.76],[0.85],[0.64],[0.67],[0.85],[0.65],[0.65],[0.66],[0.64],[0.66],[0.75],[0.67],[0.63],[0.65],[0.64]]).reshape(-1,1)
    paramax = np.copy(paramin)

    return paramin, paramax