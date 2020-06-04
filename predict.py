import numpy as np
import pandas as pd
import os
import time

import predict_save

from tqdm import tqdm
from timeit import default_timer as timer 
from plotnine import *

def estimate(x):

  """
    estimation of daily cases using inferred parameter values

    Args:
        x (array[float]): array of floats. First index is social distancing delay and 
                          the rest is seed Iu bound 
                          has to be of length (1+number of locations)
  """


  print('initial delay + seed vector:\n', x)
  sd_delay = x[0][0]
  sd_flag = True
  start_date = '2020-02-18'
  estimate_end_date = '2020-05-08'
  estimate_flag=True
  timestep = 1
  #initial bound for undocumented infected of seed locations
  np.save('initial_Iu.npy', x[1:,:])
  
  plot_attr = {'start_date': start_date,
               'base_date': '2020-03-01',
               'output_dir': 'output',
               'labels': ['SEIR Model Without Social Distancing','SEIR Model With Social Distancing','Documented New Cases'],
               'quantile_low': 0.005,
               'quantile_high': 0.995,
               'state_code': ['GE','BW','BY','BE','BB','HB','HH','HE','MV','NI','NW','RP','SL','SN','ST','SH','TH']
              }

  #estimation step
  predict_save.predict_save(plot_attr=plot_attr, 
                            sd_delay=sd_delay, 
                            sd_flag=sd_flag, 
                            estimate_flag=estimate_flag, 
                            nCPU=os.cpu_count()-1,
                            num_ens=4000, 
                            Td=6.,
                            a=1.78,
                            report_delay_bin=10, 
                            output_dir=plot_attr['output_dir']
                            )
  
def main():

  """
    set the initial values for seeds and 
    estimate daily cases using inferred parameter values
  """


  start = timer()

  cost_list=[]
  num_states = 16

  #index of seed locations on day 0 (Feb 18, 2020)
  #first variable is social distancing delay
  var_idx = np.array([0, 1, 2, 10])
  seedid = var_idx[1:]-1
  np.save('seedid.npy', seedid)
  
  #bound for documented cases of seed locations on day 0
  Id_seed = np.array([0]*num_states).reshape(-1, 1)
  Id_seed[seedid] = np.array([[1.], [1.], [1.]])
  np.save('initial_Id.npy', Id_seed*1)
  
  #bound for undocumented cases of seed locations on day 0 
  #(first variable is social distanding delay)
  x0 = np.array([0]*(num_states+1)).reshape(-1, 1)
  x0[var_idx] = np.array([[6], [200.], [200.], [400]])
  
  estimate(x0)
    
  print(timer() - start)

if __name__ == "__main__":
  main()