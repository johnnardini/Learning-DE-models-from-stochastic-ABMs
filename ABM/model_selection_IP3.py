import numpy as np

from scipy import integrate
from scipy import sparse
from scipy import interpolate

import os
import scipy.io as sio
import scipy.optimize
import itertools
import time

import pdb

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset



def g(y):
    return np.ones(y.shape)


def f(y):
    k = 1.7e-3
    return y*(1-y/k)

def learned_RHS_ODE(t,y,q,desc,deg):

    dydt = 0.0
    for i in np.arange(1,deg + 1):
        dydt += q[desc.index('C^'+str(i))]*(y**i)
    #return (q[desc.index('C^1')]*y +
    #        q[desc.index('C^2')]*y**2 +
    #        q[desc.index('C^3')]*y**3 +
    #        q[desc.index('C^4')]*y**4)
    return dydt

def learned_RHS_ODE_SIR(t,y,q,desc,deg):

    dydt = np.zeros((2,))
    
    for i,qq in enumerate(q):
                    dydt[i] = (qq[desc.index("S")]*y[0] + 
                        qq[desc.index("S^2")]*y[0]**2 + 
                        qq[desc.index("I")]*y[1] + 
                        qq[desc.index("I^2")]*y[1]**2 + 
                        qq[desc.index("IS")]*y[0]*y[1])
    
    '''for i,qq in enumerate(q):
                    dydt[i] = (qq[desc.index("I")]*y[1] + 
                        qq[desc.index("SI")]*y[0]*y[1] + 
                        qq[desc.index("I^2")]*y[1]**2 + 
                        qq[desc.index("S^2I")]*(y[0]**2)*y[1] + 
                        qq[desc.index("SI^2")]*y[0]*(y[1]**2))'''


    '''dSdt = (q[0][desc.index("S")]*y[0] + 
                        q[0][desc.index("S^2")]*y[0]**2 + 
                        q[0][desc.index("I")]*y[1] + 
                        q[0][desc.index("I^2")]*y[1]**2 + 
                        q[0][desc.index("IS")]*y[0]*y[1])
            
                dIdt = (q[1][desc.index("S")]*y[0] + 
                        q[1][desc.index("S^2")]*y[0]**2 + 
                        q[1][desc.index("I")]*y[1] + 
                        q[1][desc.index("I^2")]*y[1]**2 + 
                        q[1][desc.index("IS")]*y[0]*y[1])'''


    return dydt



def ODE_sim(q,RHS,t,IC,f=f,g=g,description=None,deg=2):
    
    #grids for numerical integration
    t_sim = np.linspace(t[0],t[-1],10000)
    
    y0 = IC
        
    #indices for integration to write to file for
    for tp in t:
        tp_ind = np.abs(tp-t_sim).argmin()
        if tp == t[0]:
            t_sim_write_ind = np.array(tp_ind)
        else:
            t_sim_write_ind = np.hstack((t_sim_write_ind,tp_ind))

    #make RHS a function of t,y
    def RHS_ty(t,y):
        return RHS(t,y,q,description,deg)
            
    y = np.zeros((len(IC),(len(t))))   # array for solution
    y[:,0] = IC
    write_count = 0

    r = integrate.ode(RHS_ty).set_integrator("dopri5")#dopri5")  # choice of method
    r.set_initial_value(y0, t[0])   # initial values
    for i in range(1, t_sim.size):
        #write to y for write indices
        if np.any(i==t_sim_write_ind):
            write_count+=1
            y[:,write_count] = r.integrate(t_sim[i])
        else:
            #otherwise just integrate
            r.integrate(t_sim[i]) # get one more value, add it to the array
        if not r.successful():
            print(q)
            print("integration failed")
            #pdb.set_trace()
            return y#1e6*np.ones(y.shape)
            #raise RuntimeError("Could not integrate")

    return y