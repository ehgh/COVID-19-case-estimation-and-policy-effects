# policy contains the information on which day the policy was introduced
# policy_lift contains information on which day the policy was relaxed
# 1000: the policy was not introduced (relaxed) in according to the file (policy/ policy_lift)
# we use data from February 18, 2020 to May 7, 2020 to estimate the coefficients 
# each number in the file policy and policy_lift files refers to the number of days from Dec 1, 2019
# - for example, Feb 15 is represented by 76

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import scipy
from scipy.stats import iqr


def RegressionModels():
    # uploading the data
    policy      = pd.read_csv("./policy.csv")
    policy_lift = pd.read_csv("./policy_lift.csv")
    popularity  = np.load("./popularity_germany.npy")
    weather     = pd.read_csv("./weather.csv")
    trend       = pd.read_csv("./trend.csv")             # cumulative trend numbers
    PTV         = pd.read_csv("./PTV.csv")

    X = []
    Y = []

    cols = ['Border Closure', 'Initial Business closure','Educational Facilities Closed',
            'Non-essential Services Closed',
            'Stay at Home Order', 'Contact Restriction',
            'Retails Closed','Trend','Tmax','PTV'] 

    # going ove 16 states
    for j in range(16):
        
        # going over the time slots in the study
        for t in range(popularity.shape[1]):
            c = []
            
            # adding the policy if yes or no
            # 79 added to t represent the shift in day number (as we count the number of days from Dec1, 2019)

            for p in range(7):      
                po = 0
                if t+79 >= policy.iloc[j,1+p]: 
                    po = 1
                if t+79 > policy_lift.iloc[j,1+p]:
                    po = 0
                c.append(po)  

            c.append(trend.iloc[t,j+1]) 
            c.append(weather.iloc[t,j+1])     
            c.append(PTV.iloc[t,1]) 

            X.append(c)
            # we store the values for three outcomes - predict commnunity mobility in parks and recreation,
            # transit stations and workplace (to estimate mobility as used in the paper, and to adjust for car and train movement)
            Y.append([popularity[j,t,0],popularity[j,t,3],popularity[j,t,4]])

    x = pd.DataFrame(X,columns=cols)
    y = pd.DataFrame(Y,columns=['mobility','train','car'])


    # Lasso Regression Models

    model_mobility = Lasso(alpha=0.25).fit(x,y['mobility'])
    model_train    = Lasso(alpha=0.25).fit(x,y['train'])
    model_car      = Lasso(alpha=0.25).fit(x,y['car'])

    # getting the coefficients
    coeff = pd.DataFrame(model2.coef_,columns=['Lasso'])
    coeff['Variables'] = cols


    # Bootstrapping

    for j in range(1000):
        # bootstrapping with 90% data
        X_train, X_test, y_train, y_test = train_test_split(x, y['mobility'], test_size=0.1, random_state=j)
        model_temp = Lasso(alpha=0.25).fit(X_train,y_train)
        coeff[str(j)] = model_temp.coef_ 

    # estimating the IQR from Bootstrapping

    IQR = np.zeros((3,10)) # 25 percentile, mean and 75th percentile for the 10 predictor variables

    for k in range(10):
        sample_standard_error = scipy.stats.sem(coeff.values[k,2:])
        q75, q25 = np.percentile(coeff.values[k,2:], [75 ,25])

        IQR[0,k] = q25
        IQR[1,k] = coeff['Lasso'].values[k]
        IQR[2,k] = q75
        
    return(model_mobility,model_train,model_car,coeff,IQR)