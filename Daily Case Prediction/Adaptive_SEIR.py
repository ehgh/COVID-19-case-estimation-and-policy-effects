import numpy as np
import scipy

def SEIR(x, M_g, M_f, pop, ts, pop0, sd=[]):

    #the Adaptive metapopulation SEIR model
    dt = 1.
    tmstep = 1
    #integrate forward for one day
    num_loc = pop.shape[0]
    (_, num_ens) = x.shape
    #S,E,Id,Iu,obs,beta,mu,theta_g,theta_f,Z,alpha,D
    Sidx = np.arange(1, 5*num_loc, 5).T
    Eidx = np.arange(2, 5*num_loc, 5).T
    Ididx = np.arange(3, 5*num_loc, 5).T
    Iuidx = np.arange(4, 5*num_loc, 5).T
    obsidx = np.arange(5, 5*num_loc+5, 5).T
    betaidx = 5*num_loc+1
    muidx = 5*num_loc+2
    thetagidx = 5*num_loc+3
    thetafidx = 5*num_loc+4
    Zidx = 5*num_loc+5
    gammaidx = 5*num_loc+6
    Didx = 5*num_loc+7
    S = np.zeros((num_loc, num_ens, tmstep+1))
    E = np.zeros((num_loc, num_ens, tmstep+1))
    Id = np.zeros((num_loc, num_ens, tmstep+1))
    Iu = np.zeros((num_loc, num_ens, tmstep+1))
    Incidence = np.zeros((num_loc, num_ens, tmstep+1))
    Incidence_u = np.zeros((num_loc, num_ens, tmstep+1))
    #initialize S,E,Id,Iu and parameters
    S[:,:,0] = x[Sidx-1,:]
    E[:,:,0] = x[Eidx-1,:]
    Id[:,:,0] = x[Ididx-1,:]
    Iu[:,:,0] = x[Iuidx-1,:]
    beta = x[betaidx-1,:].reshape(1, -1)
    mu = x[muidx-1,:].reshape(1, -1)
    theta_g = x[thetagidx-1,:].reshape(1, -1)
    theta_f = x[thetafidx-1,:].reshape(1, -1)
    Z = x[Zidx-1,:].reshape(1, -1)
    gamma = x[gammaidx-1,:].reshape(1, -1)
    D = x[Didx-1,:].reshape(1, -1)
    
    beta = np.repeat(beta, num_loc, axis=0)
    mu = np.repeat(mu, num_loc, axis=0)
    sd = np.repeat(sd[:, ts].reshape(-1, 1), num_ens, axis=1)
    gamma = np.repeat(gamma, num_loc, axis=0)
    beta = (1 + gamma * sd) * beta
     
    num_ad = num_loc
    alpha_jidx = np.arange(5*num_loc+8, 5*num_loc+8+num_ad).T
    alpha = x[alpha_jidx-1,:].reshape(num_ad, -1)

    #start integration
    tcnt = -1
    for t in np.arange(ts+1+dt, (ts+1+tmstep)+(dt), dt):
        tcnt = tcnt+1
        dt1 = dt
        #first step
        ESenter_g = dt1 * np.repeat(theta_g, num_loc, axis=0)*np.dot(M_g[:,:,ts], S[:,:,tcnt]/(pop-Id[:,:,tcnt]))
        ESleft_g = np.minimum(dt1 * np.repeat(theta_g, num_loc, axis=0)*(S[:,:,tcnt]/(pop-Id[:,:,tcnt]))*np.repeat(np.sum(M_g[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * S[:,:,tcnt])
        EEenter_g = dt1 * np.repeat(theta_g, num_loc, axis=0)*np.dot(M_g[:,:,ts], E[:,:,tcnt]/(pop-Id[:,:,tcnt]))
        EEleft_g = np.minimum(dt1 * np.repeat(theta_g, num_loc, axis=0)*(E[:,:,tcnt]/(pop-Id[:,:,tcnt]))*np.repeat(np.sum(M_g[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * E[:,:,tcnt])
        EIuenter_g = dt1 * np.repeat(theta_g, num_loc, axis=0)*np.dot(M_g[:,:,ts], Iu[:,:,tcnt]/(pop-Id[:,:,tcnt]))
        EIuleft_g = np.minimum(dt1 * np.repeat(theta_g, num_loc, axis=0)*(Iu[:,:,tcnt]/(pop-Id[:,:,tcnt]))*np.repeat(np.sum(M_g[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Iu[:,:,tcnt])
        
        ESenter_f = dt1 * np.repeat(theta_f, num_loc, axis=0)*np.dot(M_f[:,:,ts], S[:,:,tcnt]/(pop-Id[:,:,tcnt]))
        ESleft_f = np.minimum(dt1 * np.repeat(theta_f, num_loc, axis=0)*(S[:,:,tcnt]/(pop-Id[:,:,tcnt]))*np.repeat(np.sum(M_f[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * S[:,:,tcnt])
        EEenter_f = dt1 * np.repeat(theta_f, num_loc, axis=0)*np.dot(M_f[:,:,ts], E[:,:,tcnt]/(pop-Id[:,:,tcnt]))
        EEleft_f = np.minimum(dt1 * np.repeat(theta_f, num_loc, axis=0)*(E[:,:,tcnt]/(pop-Id[:,:,tcnt]))*np.repeat(np.sum(M_f[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * E[:,:,tcnt])
        EIuenter_f = dt1 * np.repeat(theta_f, num_loc, axis=0)*np.dot(M_f[:,:,ts], Iu[:,:,tcnt]/(pop-Id[:,:,tcnt]))
        EIuleft_f = np.minimum(dt1 * np.repeat(theta_f, num_loc, axis=0)*(Iu[:,:,tcnt]/(pop-Id[:,:,tcnt]))*np.repeat(np.sum(M_f[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Iu[:,:,tcnt])
        
        Eexpd = dt1 * beta*S[:,:,tcnt]*Id[:,:,tcnt]/pop
        Eexpu = dt1 * mu*beta*S[:,:,tcnt]*Iu[:,:,tcnt]/pop
        Einfd = dt1 * alpha*E[:,:,tcnt]/Z
        Einfu = dt1 * (1.-alpha)*E[:,:,tcnt]/Z
        Erecd = dt1 * Id[:,:,tcnt]/D
        Erecu = dt1 * Iu[:,:,tcnt]/D

        ESenter_g[ESenter_g<0] = 0.
        ESleft_g[ESleft_g<0] = 0.
        EEenter_g[EEenter_g<0] = 0.
        EEleft_g[EEleft_g<0] = 0.
        EIuenter_g[EIuenter_g<0] = 0.
        EIuleft_g[EIuleft_g<0] = 0.
        ESenter_f[ESenter_f<0] = 0.
        ESleft_f[ESleft_f<0] = 0.
        EEenter_f[EEenter_f<0] = 0.
        EEleft_f[EEleft_f<0] = 0.
        EIuenter_f[EIuenter_f<0] = 0.
        EIuleft_f[EIuleft_f<0] = 0.
        Eexpd[Eexpd<0] = 0.
        Eexpu[Eexpu<0] = 0.
        Einfd[Einfd<0] = 0.
        Einfu[Einfu<0] = 0.
        Erecd[Erecd<0] = 0.
        Erecu[Erecu<0] = 0.

        sk1 = -Eexpd-Eexpu+ESenter_g-ESleft_g+ESenter_f-ESleft_f
        ek1 = Eexpd+Eexpu-Einfd-Einfu+EEenter_g-EEleft_g+EEenter_f-EEleft_f
        idk1 = Einfd-Erecd
        iuk1 = Einfu-Erecu+EIuenter_g-EIuleft_g+EIuenter_f-EIuleft_f
        ik1i = Einfd
        ik1i_u = Einfu
        
        #second step
        Ts1 = S[:,:,tcnt]+sk1/2.
        Te1 = E[:,:,tcnt]+ek1/2.
        Tis1 = Id[:,:,tcnt]+idk1/2.
        Tia1 = Iu[:,:,tcnt]+iuk1/2.

        ESenter_g = dt1 * np.repeat(theta_g, num_loc, axis=0)*np.dot(M_g[:,:,ts], Ts1/(pop-Tis1))
        ESleft_g = np.minimum(dt1 * np.repeat(theta_g, num_loc, axis=0)*(Ts1/(pop-Tis1))*np.repeat(np.sum(M_g[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Ts1)
        EEenter_g = dt1 * np.repeat(theta_g, num_loc, axis=0)*np.dot(M_g[:,:,ts], Te1/(pop-Tis1))
        EEleft_g = np.minimum(dt1 * np.repeat(theta_g, num_loc, axis=0)*(Te1/(pop-Tis1))*np.repeat(np.sum(M_g[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Te1)
        EIuenter_g = dt1 * np.repeat(theta_g, num_loc, axis=0)*np.dot(M_g[:,:,ts], Tia1/(pop-Tis1))
        EIuleft_g = np.minimum(dt1 * np.repeat(theta_g, num_loc, axis=0)*(Tia1/(pop-Tis1))*np.repeat(np.sum(M_g[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Tia1)
        
        ESenter_f = dt1 * np.repeat(theta_f, num_loc, axis=0)*np.dot(M_f[:,:,ts], Ts1/(pop-Tis1))
        ESleft_f = np.minimum(dt1 * np.repeat(theta_f, num_loc, axis=0)*(Ts1/(pop-Tis1))*np.repeat(np.sum(M_f[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Ts1)
        EEenter_f = dt1 * np.repeat(theta_f, num_loc, axis=0)*np.dot(M_f[:,:,ts], Te1/(pop-Tis1))
        EEleft_f = np.minimum(dt1 * np.repeat(theta_f, num_loc, axis=0)*(Te1/(pop-Tis1))*np.repeat(np.sum(M_f[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Te1)
        EIuenter_f = dt1 * np.repeat(theta_f, num_loc, axis=0)*np.dot(M_f[:,:,ts], Tia1/(pop-Tis1))
        EIuleft_f = np.minimum(dt1 * np.repeat(theta_f, num_loc, axis=0)*(Tia1/(pop-Tis1))*np.repeat(np.sum(M_f[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Tia1)
        
        Eexpd = dt1 * beta*Ts1*Tis1/pop
        Eexpu = dt1 * mu*beta*Ts1*Tia1/pop
        Einfd = dt1 * alpha*Te1/Z
        Einfu = dt1 * (1.-alpha)*Te1/Z
        Erecd = dt1 * Tis1/D
        Erecu = dt1 * Tia1/D
        
        ESenter_g[ESenter_g<0] = 0.
        ESleft_g[ESleft_g<0] = 0.
        EEenter_g[EEenter_g<0] = 0.
        EEleft_g[EEleft_g<0] = 0.
        EIuenter_g[EIuenter_g<0] = 0.
        EIuleft_g[EIuleft_g<0] = 0.
        ESenter_f[ESenter_f<0] = 0.
        ESleft_f[ESleft_f<0] = 0.
        EEenter_f[EEenter_f<0] = 0.
        EEleft_f[EEleft_f<0] = 0.
        EIuenter_f[EIuenter_f<0] = 0.
        EIuleft_f[EIuleft_f<0] = 0.
        Eexpd[Eexpd<0] = 0.
        Eexpu[Eexpu<0] = 0.
        Einfd[Einfd<0] = 0.
        Einfu[Einfu<0] = 0.
        Erecd[Erecd<0] = 0.
        Erecu[Erecu<0] = 0.
        
        sk2 = -Eexpd-Eexpu+ESenter_g-ESleft_g+ESenter_f-ESleft_f
        ek2 = Eexpd+Eexpu-Einfd-Einfu+EEenter_g-EEleft_g+EEenter_f-EEleft_f
        idk2 = Einfd-Erecd
        iuk2 = Einfu-Erecu+EIuenter_g-EIuleft_g+EIuenter_f-EIuleft_f
        ik2i = Einfd
        ik2i_u = Einfu

        #third step
        Ts2 = S[:,:,tcnt]+sk2/2.
        Te2 = E[:,:,tcnt]+ek2/2.
        Tis2 = Id[:,:,tcnt]+idk2/2.
        Tia2 = Iu[:,:,tcnt]+iuk2/2.

        ESenter_g = dt1 * np.repeat(theta_g, num_loc, axis=0)*np.dot(M_g[:,:,ts], Ts2/(pop-Tis2))
        ESleft_g = np.minimum(dt1 * np.repeat(theta_g, num_loc, axis=0)*(Ts2/(pop-Tis2))*np.repeat(np.sum(M_g[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Ts2)
        EEenter_g = dt1 * np.repeat(theta_g, num_loc, axis=0)*np.dot(M_g[:,:,ts], Te2/(pop-Tis2))
        EEleft_g = np.minimum(dt1 * np.repeat(theta_g, num_loc, axis=0)*(Te2/(pop-Tis2))*np.repeat(np.sum(M_g[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Te2)
        EIuenter_g = dt1 * np.repeat(theta_g, num_loc, axis=0)*np.dot(M_g[:,:,ts], Tia2/(pop-Tis2))
        EIuleft_g = np.minimum(dt1 * np.repeat(theta_g, num_loc, axis=0)*(Tia2/(pop-Tis2))*np.repeat(np.sum(M_g[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Tia2)
        
        ESenter_f = dt1 * np.repeat(theta_f, num_loc, axis=0)*np.dot(M_f[:,:,ts], Ts2/(pop-Tis2))
        ESleft_f = np.minimum(dt1 * np.repeat(theta_f, num_loc, axis=0)*(Ts2/(pop-Tis2))*np.repeat(np.sum(M_f[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Ts2)
        EEenter_f = dt1 * np.repeat(theta_f, num_loc, axis=0)*np.dot(M_f[:,:,ts], Te2/(pop-Tis2))
        EEleft_f = np.minimum(dt1 * np.repeat(theta_f, num_loc, axis=0)*(Te2/(pop-Tis2))*np.repeat(np.sum(M_f[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Te2)
        EIuenter_f = dt1 * np.repeat(theta_f, num_loc, axis=0)*np.dot(M_f[:,:,ts], Tia2/(pop-Tis2))
        EIuleft_f = np.minimum(dt1 * np.repeat(theta_f, num_loc, axis=0)*(Tia2/(pop-Tis2))*np.repeat(np.sum(M_f[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Tia2)
        
        Eexpd = dt1 * beta*Ts2*Tis2/pop
        Eexpu = dt1 * mu*beta*Ts2*Tia2/pop
        Einfd = dt1 * alpha*Te2/Z
        Einfu = dt1 * (1.-alpha)*Te2/Z
        Erecd = dt1 * Tis2/D
        Erecu = dt1 * Tia2/D

        ESenter_g[ESenter_g<0] = 0.
        ESleft_g[ESleft_g<0] = 0.
        EEenter_g[EEenter_g<0] = 0.
        EEleft_g[EEleft_g<0] = 0.
        EIuenter_g[EIuenter_g<0] = 0.
        EIuleft_g[EIuleft_g<0] = 0.
        ESenter_f[ESenter_f<0] = 0.
        ESleft_f[ESleft_f<0] = 0.
        EEenter_f[EEenter_f<0] = 0.
        EEleft_f[EEleft_f<0] = 0.
        EIuenter_f[EIuenter_f<0] = 0.
        EIuleft_f[EIuleft_f<0] = 0.
        Eexpd[Eexpd<0] = 0.
        Eexpu[Eexpu<0] = 0.
        Einfd[Einfd<0] = 0.
        Einfu[Einfu<0] = 0.
        Erecd[Erecd<0] = 0.
        Erecu[Erecu<0] = 0.

        sk3 = -Eexpd-Eexpu+ESenter_g-ESleft_g+ESenter_f-ESleft_f
        ek3 = Eexpd+Eexpu-Einfd-Einfu+EEenter_g-EEleft_g+EEenter_f-EEleft_f
        idk3 = Einfd-Erecd
        iuk3 = Einfu-Erecu+EIuenter_g-EIuleft_g+EIuenter_f-EIuleft_f
        ik3i = Einfd
        ik3i_u = Einfu

        #fourth step
        Ts3 = S[:,:,tcnt]+sk3
        Te3 = E[:,:,tcnt]+ek3
        Tis3 = Id[:,:,tcnt]+idk3
        Tia3 = Iu[:,:,tcnt]+iuk3

        ESenter_g = dt1 * np.repeat(theta_g, num_loc, axis=0)*np.dot(M_g[:,:,ts], Ts3/(pop-Tis3))
        ESleft_g = np.minimum(dt1 * np.repeat(theta_g, num_loc, axis=0)*(Ts3/(pop-Tis3))*np.repeat(np.sum(M_g[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Ts3)
        EEenter_g = dt1 * np.repeat(theta_g, num_loc, axis=0)*np.dot(M_g[:,:,ts], Te3/(pop-Tis3))
        EEleft_g = np.minimum(dt1 * np.repeat(theta_g, num_loc, axis=0)*(Te3/(pop-Tis3))*np.repeat(np.sum(M_g[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Te3)
        EIuenter_g = dt1 * np.repeat(theta_g, num_loc, axis=0)*np.dot(M_g[:,:,ts], Tia3/(pop-Tis3))
        EIuleft_g = np.minimum(dt1 * np.repeat(theta_g, num_loc, axis=0)*(Tia3/(pop-Tis3))*np.repeat(np.sum(M_g[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Tia3)
        
        ESenter_f = dt1 * np.repeat(theta_f, num_loc, axis=0)*np.dot(M_f[:,:,ts], Ts3/(pop-Tis3))
        ESleft_f = np.minimum(dt1 * np.repeat(theta_f, num_loc, axis=0)*(Ts3/(pop-Tis3))*np.repeat(np.sum(M_f[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Ts3)
        EEenter_f = dt1 * np.repeat(theta_f, num_loc, axis=0)*np.dot(M_f[:,:,ts], Te3/(pop-Tis3))
        EEleft_f = np.minimum(dt1 * np.repeat(theta_f, num_loc, axis=0)*(Te3/(pop-Tis3))*np.repeat(np.sum(M_f[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Te3)
        EIuenter_f = dt1 * np.repeat(theta_f, num_loc, axis=0)*np.dot(M_f[:,:,ts], Tia3/(pop-Tis3))
        EIuleft_f = np.minimum(dt1 * np.repeat(theta_f, num_loc, axis=0)*(Tia3/(pop-Tis3))*np.repeat(np.sum(M_f[:,:,ts], 0, keepdims=True).T, num_ens, axis=1), dt1 * Tia3)
        
        Eexpd = dt1 * beta*Ts3*Tis3/pop
        Eexpu = dt1 * mu*beta*Ts3*Tia3/pop
        Einfd = dt1 * alpha*Te3/Z
        Einfu = dt1 * (1.-alpha)*Te3/Z
        Erecd = dt1 * Tis3/D
        Erecu = dt1 * Tia3/D
        
        ESenter_g[ESenter_g<0] = 0.
        ESleft_g[ESleft_g<0] = 0.
        EEenter_g[EEenter_g<0] = 0.
        EEleft_g[EEleft_g<0] = 0.
        EIuenter_g[EIuenter_g<0] = 0.
        EIuleft_g[EIuleft_g<0] = 0.
        ESenter_f[ESenter_f<0] = 0.
        ESleft_f[ESleft_f<0] = 0.
        EEenter_f[EEenter_f<0] = 0.
        EEleft_f[EEleft_f<0] = 0.
        EIuenter_f[EIuenter_f<0] = 0.
        EIuleft_f[EIuleft_f<0] = 0.
        Eexpd[Eexpd<0] = 0.
        Eexpu[Eexpu<0] = 0.
        Einfd[Einfd<0] = 0.
        Einfu[Einfu<0] = 0.
        Erecd[Erecd<0] = 0.
        Erecu[Erecu<0] = 0.

        sk4 = -Eexpd-Eexpu+ESenter_g-ESleft_g+ESenter_f-ESleft_f
        ek4 = Eexpd+Eexpu-Einfd-Einfu+EEenter_g-EEleft_g+EEenter_f-EEleft_f
        idk4 = Einfd-Erecd
        iuk4 = Einfu-Erecu+EIuenter_g-EIuleft_g+EIuenter_f-EIuleft_f
        ik4i = Einfd
        ik4i_u = Einfu

        #####
        S[:,:,tcnt+1] = S[:,:,tcnt]+np.round((sk1/6.+sk2/3.+sk3/3.+sk4/6.))
        E[:,:,tcnt+1] = E[:,:,tcnt]+np.round((ek1/6.+ek2/3.+ek3/3.+ek4/6.))
        Id[:,:,tcnt+1] = Id[:,:,tcnt]+np.round((idk1/6.+idk2/3.+idk3/3.+idk4/6.))
        Iu[:,:,tcnt+1] = Iu[:,:,tcnt]+np.round((iuk1/6.+iuk2/3.+iuk3/3.+iuk4/6.))
        Incidence[:,:,tcnt+1] = np.round((ik1i/6.+ik2i/3.+ik3i/3.+ik4i/6.))
        Incidence_u[:,:,tcnt+1] = np.round((ik1i_u/6.+ik2i_u/3.+ik3i_u/3.+ik4i_u/6.))

    ###update x
    x[Sidx-1,:] = S[:,:,tcnt+1]
    x[Eidx-1,:] = E[:,:,tcnt+1]
    x[Ididx-1,:] = Id[:,:,tcnt+1]
    x[Iuidx-1,:] = Iu[:,:,tcnt+1]
    x[obsidx-1,:] = Incidence[:,:,tcnt+1]
    ###update pop
    pop = pop-np.sum(M_g[:,:,ts], 0, keepdims=True).T * theta_g + np.sum(M_g[:,:,ts], 1, keepdims=True) * theta_g -\
            np.sum(M_f[:,:,ts], 0, keepdims=True).T * theta_f + np.sum(M_f[:,:,ts], 1, keepdims=True) * theta_f
    minfrac = 0.6
    pop[pop<np.dot(minfrac, pop0)] = np.dot(pop0[pop<np.dot(minfrac, pop0)], minfrac)
    
    return x, pop, Incidence[:,:,tcnt+1], Incidence_u[:,:,tcnt+1]