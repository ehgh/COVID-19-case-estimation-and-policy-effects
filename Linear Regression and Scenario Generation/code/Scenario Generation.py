def Scenario_Generation():


    # first restricting the data to April 2020 when we are predicting six weeks out from april 2020
    popularity_germany = np.load("./popularity_germany.npy")
    popularity_germany = np.copy(popularity_germany[:,0:63,:]) # april 20th is the 63rd index in the popularity number
    # 
    one = np.multiply(np.ones((16,42,6)),popularity_germany[:,62:63,:])
    popularity_germany = np.append(popularity_germany,one,axis=1)

    # bus movement is kept as 0 after march 18
    # we keep the flight data same as what was observed on april 20th
    # there has been no change in the trucj movement pattern (sp we keep as a weekly pattern)
    # the data contains placeholder for 158 countries (so that we can capture the movement from all the countries)

    bus_movement    = np.load("./bus_movement.npy")
    truck_movement  = np.load("./truck_movement.npy")
    flight_movement = np.load("./flight_movement.npy")
    car_movement    = np.load("./car_movement.npy")
    train_movement  = np.load("./train_movement.npy")

    bus_move    = bus_movement[:,0:63,:]
    truck_move  = truck_movement[:,0:63,:]
    flight_move = flight_movement[:,0:63,:]
    static_car_move    = car_movement[:,0:63,:]
    static_train_move  = train_movement[:,0:63,:]

    one = np.multiply(np.ones((16,42,158)),bus_move[:,62:63,:])
    bus_move = np.append(bus_move,one,axis=1)
    one = np.multiply(np.ones((16,42,158)),truck_move[:,62:63,:])
    truck_move = np.append(truck_move,one,axis=1)
    one = np.multiply(np.ones((16,42,158)),flight_move[:,62:63,:])
    flight_move = np.append(flight_move,one,axis=1)

    one = np.multiply(np.ones((16,42,158)),static_car_move[:,62:63,:])
    static_car_move = np.append(static_car_move,one,axis=1)
    one = np.multiply(np.ones((16,42,158)),static_train_move[:,62:63,:])
    static_train_move = np.append(static_train_move,one,axis=1)

    for t in range(63,63+42):
        truck_move[:,t,:] = truck_move[:,t-7,:]

    popularity_o = np.copy(popularity_germany)

    policy_o = pd.read_csv("./policy.csv")
    policy_o_life = pd.read_csv("./policy_lift.csv")

    cols = ['Border Closure', 'Initial business closure',
           'Educational facilities closed', 'Non-essential services closed',
           'Stay at home order', 'contact restriction',
           'retails closed','trend','tmax','frustration']


    policy      = pd.read_csv("./policy.csv")
    policy_lift = pd.read_csv("./policy_lift.csv")
    popularity  = np.load("./popularity_germany.npy")
    weather     = pd.read_csv("./weather_predict.csv")
    trend       = pd.read_csv("./trend_predict.csv")             # cumulative trend numbers
    PTV         = pd.read_csv("./PTV_predict.csv")

    # there are 9 scenarios
    # in first scenario, no change in policy and all policies remain in place
    # in the next 7 scenarios, we switch off (relax) one of the policy if it was implemented in that respective state
    # in the last scenario, we relax all the policies (however, we do not show this in paper as it lead to a very sharp rise)
    # each of these 9 scenarios is tested twice - one when for april 21, 2020 and once for april 28, 2020

    for pp in range(9):
        popularity = np.copy(popularity_o)

        car_movement   = np.copy(static_car_move)
        train_movement = np.copy(static_train_move)

        for w in range(2):
            policy1 = pd.DataFrame.copy(pd.read_csv("./policy.csv")  )
            policy2 = pd.DataFrame.copy(pd.read_csv("./policy.csv")  )
            policy3 = pd.DataFrame.copy(pd.read_csv("./policy.csv")  )

            name = '_P_'+str(pp)+'_W_'+str(w+1)

            if pp == 0:
                name = ''

            # when relaxing a policy, we keep the date very high (1000) so that the return value is 0 (not implemented post april 21 or april 28)
            elif pp == 8:
                if w == 0:
                    name = '_P_8_W_1'
                    for xx in range(7):
                        policy1[cols[xx]] = 1000
                        policy2[cols[xx]] = 1000
                        policy3[cols[xx]] = 1000

                if w == 1:
                    name = '_P_8_W_2'
                    for xx in range(7):
                        policy2[cols[xx]] = 1000
                        policy3[cols[xx]] = 1000
            else:
                if w == 0:
                    policy1[cols[pp-1]] = 1000
                    policy2[cols[pp-1]] = 1000
                    policy3[cols[pp-1]] = 1000
                elif w==1:
                    policy2[cols[pp-1]] = 1000
                    policy3[cols[pp-1]] = 1000

            X = []

            for j in range(16):

                # first week
                for t in range(63,70):
                    c = []
                    for p in range(7):
                        c.append(int(policy1[cols[p]].iloc[j] <= t+79)) 

                    c.append(trend.iloc[t,j+1]) 
                    c.append(weather.iloc[t,j+1])     
                    c.append(PTV.iloc[t,1]) 

                    X.append(c)

                # second week
                for t in range(70,77):
                    c = []
                    for p in range(7): 
                        c.append(int(policy2[cols[p]].iloc[j] <= t+79)) 

                    c.append(trend.iloc[t,j+1]) 
                    c.append(weather.iloc[t,j+1])     
                    c.append(PTV.iloc[t,1])  

                    X.append(c)

                # rest of the four weeks
                for ww in range(4):
                    for t in range(77+7*ww,84+7*ww):
                        c = []
                        for p in range(7):
                            c.append(int(policy3[cols[p]].iloc[j] <= t+79)) 

                        c.append(trend.iloc[t,j+1]) 
                        c.append(weather.iloc[t,j+1])     
                        c.append(PTV.iloc[t,1]) 

                        X.append(c)


            x = pd.DataFrame(X,columns=cols)

            models = RegressionModels()

            y_pred   = models[0].predict(x)
            y_car    = models[1].predict(x)
            y_train  = models[2].predict(x)


            for j in range(16):
                for t in range(63,63+42):
                    popularity[j,t,0] = y_pred[42*j+t-63]
                    popularity[j,t,3] = y_car[42*j+t-63]
                    popularity[j,t,4] = y_train[42*j+t-63]

            wtrain = popularity[:,:,3]
            wtrain = (wtrain)/100+1

            wcars = popularity[:,:,4]
            wcars = (wcars)/100+1

            numbee = 63+42
            for i in range(16):
                car_movement[i] = np.multiply(car_movement[i],wcars[i].reshape([numbee,1])*np.ones([numbee,158]))
                train_movement[i] = np.multiply(train_movement[i],wtrain[i].reshape([numbee,1])*np.ones([numbee,158]))

            # the files can be saved
            #np.save('popularity_germany'+name,popularity)
            #np.save('train_movement'+name,train_movement)
            #np.save('car_movement'+name,car_movement)
            
    return()