import numpy as np
import scipy
import scipy.io as sio
import sys
import pandas as pd
import os
import datetime
import time
import copy

import estimator

from tqdm import tqdm
from timeit import default_timer as timer 
from initialize import initialize
from Adaptive_SEIR import SEIR
from multiprocessing import Pool
from plotnine import *
from mizani.breaks import date_breaks
from mizani.formatters import date_format
from itertools import repeat

np.set_printoptions(threshold = sys.maxsize, precision = 10, suppress=True)

def reject_outliers(df, group='time', col='value', low=0.05, high=0.95):
    
    df = df[df.groupby(group)[col].\
      transform(lambda x : (x<x.quantile(high))&(x>(x.quantile(low)))).eq(1)]
    
    return df

def numpy2pandas(data, start_date='2020-2-19', base_date='2020-03-01'):

    """
    convert a numpy array to panas dataframe

    Args:
        data (array): numpy array, last dimension must be time
        start_date (str): staring date in numpy array
                          format: '%Y-%m-%d'      
        base_date (str): base date to filter the dataframe
                          format: '%Y-%m-%d'      

    Returns:
        df (dataframe): dataframe with column `time` between start_date and base_date
                        and column `value` for numpy array values
                        columns = ['time', 'value']
    """
    
    time_seq_start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    time_seq = pd.date_range(time_seq_start, periods=data.shape[-1], freq='d')
    
    df = pd.DataFrame(data, columns=time_seq)
    df = df.melt(var_name='time', value_name='daily_cases')
    df = df.sort_values('time')
    df['time'] = df['time'].dt.strftime('%Y-%m-%d')
    
    df = df[df['time'] > base_date]

    return df
    
def estimate_save(data, plot_attr, sd, plot_type, rej_out=False):

    for state in range(data.shape[0]):
        df = numpy2pandas(data[state,:,:], start_date=plot_attr['start_date'], base_date=plot_attr['base_date'])
        df['sd'] = sd
        df['plot_type'] = plot_type
        if rej_out and sd != -1:
            df = reject_outliers(df, col='daily_cases', low=plot_attr['quantile_low'], high=plot_attr['quantile_high'])
        df.to_pickle(os.path.join(plot_attr['output_dir'],'daily_cases_est_sd_'+sd+'_state_'+plot_attr['state_code'][state+1]+'.pkl'))
        if sd == '1':
          print('Estimated daily cases of state {} with social distancing\n'.format(plot_attr['state_code'][state+1]), df)
        elif sd == '0':
          print('Estimated daily cases of state {} without social distancing\n'.format(plot_attr['state_code'][state+1]), df)
        else:
          print('Daily cases of state {} \n'.format(plot_attr['state_code'][state+1]), df)

    #sum over all states
    df = numpy2pandas(data.sum(axis=0), start_date=plot_attr['start_date'], base_date=plot_attr['base_date'])
    df['sd'] = sd
    df['plot_type'] = plot_type
    df.to_pickle(os.path.join(plot_attr['output_dir'],'daily_cases_est_sd_'+sd+'_state_'+plot_attr['state_code'][0]+'.pkl'))
    if sd == '1':
      print('Estimated daily cases of Germany with social distancing\n', df)
    elif sd == '0':
      print('Estimated daily cases of Germany without social distancing\n', df)
    else:
      print('Daily cases of Germany \n', df)
    
def predict_save(plot_attr,
                sd_delay=0, 
                estimate_flag=False,
                nCPU=os.cpu_count()-7,
                num_ens=4000, 
                Td=6.,
                a=1.78,
                report_delay_bin=10, 
                output_dir=time.strftime("%Y%m%d-%H%M%S"),
                sd_flag=True
                ):

    """
    estimation of daily cases using set parameter values for 
    both cases of with and without social distancing and save output 
    for each location

    Args:
        plot_attr (dict): attributes of estimation (refer to predict.py for more details)
        sd_delay (int): number of days to shift social distancing 
        estimate_flag (bool): flag to use estimate daily cases
        nCPU (int): number of cores to use for multiprocessing
        num_ens (int): number of ensembles to estimate daily cases
        Td (float): average reporting delay of documented cases
        a (float): shape parameter of gamma distribution of reporting delay, Td=a*b
        report_delay_bin (int): bin size of daily cases to add reporting delay (per state/ensemble) > 0
        output_dir (str): directory to save output files, default is current date and time
        sd_flag (bool): flag to use social distancing in SEIR model           
    
    """

    start = timer()
    df = None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args = [num_ens, Td, a, report_delay_bin, sd_delay, output_dir]

    if estimate_flag:
        #daily cases with social distancing
        obs_temp, daily_cases_u, x, pop = estimator.estimator(*args, sd_flag=True)
        np.save(os.path.join('obs_temp_sd_1.npy'), obs_temp)
        estimate_save(data=obs_temp, plot_attr=plot_attr, sd='1', plot_type='boxplot', rej_out=True)
        
        #daily cases without social distancing
        if 'SEIR Model Without Social Distancing' in plot_attr['labels']:
            args[3] = 15
            obs_temp, daily_cases_u, x, pop = estimator.estimator(*args, sd_flag=False)
            np.save(os.path.join('obs_temp_sd_0.npy'), obs_temp)
            estimate_save(data=obs_temp, plot_attr=plot_attr, sd='0', plot_type='boxplot', rej_out=True)

        #ground truth daily cases     
        daily_cases_ground_truth = np.load('incidence.npy')[0:obs_temp.shape[-1],:]
        estimate_save(data=daily_cases_ground_truth.T[:, None, :], plot_attr=plot_attr, sd='-1', plot_type='point')
    
