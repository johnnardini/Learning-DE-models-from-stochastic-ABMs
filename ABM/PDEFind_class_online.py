import numpy as np
import pdb
import os
import time 
from collections import Counter

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.signal import savgol_filter

from PDE_FIND3 import *
from model_selection_IP3 import *
import imageio

class PDE_Findclass(object):
    
    def __init__(self,
                 data_file,
                 comp_str,
                 model_name,
                 trainPerc=0.5,
                 valPerc=0.5,
                 data_dir = "data/",
                 write_dir = "pickle_data/",
                 algo_name = "Greedy",
                 shuf_method = "perm",
                 prune_level = 0.05,
                 deg = 2,
                 reals = 200,
                 deriv_index = None,
                 print_pdes = False,
                 save_xi = False,
                 save_learned_xi = False,
                 num_eqns = 3,
                 save_learned_eqns=False,
                 animations = 'animations',
		         dims=1,
                 ):

        self.data_file = data_file
        #where data located
        self.data_dir = [data_dir + d  + comp_str for d in data_file]
        #where analytical data located
        self.true_data_dir = [data_dir + d for d in data_file]
        #where to write results
        self.write_dir = [write_dir + algo_name +'_' + d + '_' + model_name 
                          + '_deg_' + str(deg) for d in data_file]
        
        self.comp_str = comp_str
        
        if not os.path.exists(write_dir):
            print("creating file " + write_dir)
            os.makedirs(write_dir)
        
        self.animations = animations
        if not os.path.exists(animations):
            print("creating file " + animations)
            os.makedirs(animations)
            os.makedirs(animations + "/gifs")
        
        
        #for PDE-FIND implementation
        self.algo_name = algo_name
        self.trainPerc = trainPerc
        self.valPerc = valPerc
        self.prune_level = prune_level
        self.deg = deg
        self.reals = reals
        self.shuf_method = shuf_method
        if deriv_index == None:
            self.deriv_index = [None for d in self.data_dir]
        else:
            self.deriv_index = deriv_index
        self.print_pdes = print_pdes
        self.save_xi = save_xi
        self.save_learned_xi = save_learned_xi
        self.num_eqns = num_eqns
        self.save_learned_eqns = save_learned_eqns
        self.dims = dims
        
    def train_val_PDEFind(self):
        
        #create list of final results
        xi_list = [[] for d in self.data_dir]
        xi_list_no_prune = [[] for d in self.data_dir]
        data_description = [chr(117+i) for i in np.arange(len(self.data_dir))]
        
        #load in data source
        mat = [np.load(d+'.npy',allow_pickle=True, encoding='latin1').item() for d in self.data_dir]


        if self.dims == 1:        
            #load in library and du/dt
            X,T,Ut,Theta,description = theta_construct_two_subpops(mat,self.deg)
        elif self.dims == 2:
            X,Y,T,Ut,Theta,description = theta_construct_2d(mat,self.deg) 
    
        
        self.description = description
        x = np.unique(X)
        t = np.unique(T)
        if self.dims == 2:
            y = np.unique(Y)

        #(optional, but helps in some cases) remove the column of 1's from Theta, desctipion 
        Theta = np.delete(Theta,description.index(""),axis=1)
        self.description.remove("")
        
        #for debugging
        #self.X = X
        #self.T = T
        #self.Ut = Ut
        #self.Theta = Theta
        
        
        #now perform the training "reals" times
        for r in np.arange(self.reals):
            for i,d in enumerate(self.data_dir):
                
                #shuffle ut values into train-val data
                if self.shuf_method == 'neighbors_1d':
                    utTrain,thetaTrain,ptrain,utVal,thetaVal,pval = data_shuf_1d(Ut[i],Theta,self.trainPerc,self.valPerc,X,T,N_neighbors = 9)
                elif self.shuf_method == 'neighbors_2d':
                    utTrain,thetaTrain,ptrain,utVal,thetaVal,pval = data_shuf_2d(Ut[i],Theta,self.trainPerc,self.valPerc,X,Y,T,N_neighbors = 27)
                else:    
                    utTrain,thetaTrain,ptrain,utVal,thetaVal,pval,_,_,_ = data_shuf(Ut[i],Theta,self.shuf_method,self.trainPerc,self.valPerc,len(x),len(t))
                
		             
                xi, hparams, val_score,num_nonzero_u = run_PDE_Find_train_val(thetaTrain,
                                            utTrain,
                                            thetaVal,
                                            utVal,
                                            self.algo_name,
                                            self.description,
                                            lambda_lb=-3,
                                            lambda_ub=-1,
                                            deriv_list=self.deriv_index[i])
                
                val_score_0 = np.min(val_score)

                #perform pruning
                if len(xi[xi!=0]) > 1:
                    xi_new, val_score_new = PDE_FIND_prune_lstsq(xi,utTrain,utVal,thetaTrain,thetaVal,self.description,val_score_0,self.prune_level)
                else:
                    #don't prune if xi only have one nonzero vector
                    xi_new = xi
                
                if self.print_pdes == True:
                    print("Predicted equation is " + print_pde(xi_new,self.description,ut=data_description[i]+'_{t}'))
                xi_list_no_prune[i].append(xi)
                xi_list[i].append(xi_new)
                
                #save results?
                if self.save_xi == True:
                    np.savez(self.write_dir[i],
                    xi_list = xi_list[i],
                    xi_list_no_prune=xi_list_no_prune[i],
                    description=self.description)

 
    def train_val_ODEFind(self):
        
        #create list of final results
        xi_list = [[] for d in self.data_dir]
        xi_list_no_prune = [[] for d in self.data_dir]
        data_description = [chr(117+i) for i in np.arange(len(self.data_dir))]
        
        #load in data source
        mat = [np.load(d+'.npy',allow_pickle=True, encoding='latin1').item() for d in self.data_dir]
        
        
        T = mat[0]['variables'][0]
        U = mat[0]['variables'][1][:,0]
        Ut = [mat[0]['variables'][2]]

        try:
            F = mat[0]['F']
            self.F = F
        except:
            pass

        #Theta = [U**i for i in np.arange(self.deg)]
        Theta = np.zeros((len(T),self.deg+1))
        for i in np.arange(self.deg + 1):
            Theta[:,i] = U**i
        description = ['C^'+str(i) for i in np.arange(self.deg+1)]
        description[0] = '1'

        #(optional, but helps in some cases) remove the column of 1's from Theta, desctipion 
        Theta = np.delete(Theta,description.index("1"),axis=1)
        description.remove("1")
        
        
        self.description = description
        self.y = U
        
        
        t = np.unique(T)
        x = np.array(1,ndmin=1)
        
        print("Running " + str(self.reals) + " SinDy Simulations to determine learned equation")
        print("library is " + str(description))

        self.t = t

        #now perform the training "reals" times
        for r in np.arange(self.reals):
            for i,d in enumerate(self.data_dir):
                
                #shuffle ut values into train-val data
                if self.shuf_method == 'neighbors_1d':
                    utTrain,thetaTrain,ptrain,utVal,thetaVal,pval = data_shuf_1d(Ut[i],Theta,self.trainPerc,self.valPerc,X,T,N_neighbors = 9)
                elif self.shuf_method == 'neighbors_2d':
                    utTrain,thetaTrain,ptrain,utVal,thetaVal,pval = data_shuf_2d(Ut[i],Theta,self.trainPerc,self.valPerc,X,Y,T,N_neighbors = 27)
                else:    
                    utTrain,thetaTrain,ptrain,utVal,thetaVal,pval,_,_,_ = data_shuf(Ut[i],Theta,self.shuf_method,self.trainPerc,self.valPerc,len(x),len(t))
                
                xi, hparams, val_score,num_nonzero_u = run_PDE_Find_train_val(thetaTrain,
                                            utTrain,
                                            thetaVal,
                                            utVal,
                                            self.algo_name,
                                            self.description,
                                            lambda_lb=-6, #-4 , -8
                                            lambda_ub=-2, #-1 , -4
                                            deriv_list=self.deriv_index[i])

                val_score_0 = np.min(val_score)

                #perform pruning
                if len(xi[xi!=0]) > 1:
                    xi_new, val_score_new = PDE_FIND_prune_lstsq(xi,
                                                                 utTrain,
                                                                 utVal,
                                                                 thetaTrain,
                                                                 thetaVal,
                                                                 self.description,
                                                                 val_score_0,
                                                                 self.prune_level)
                else:
                    #don't prune if xi only have one nonzero vector
                    xi_new = xi
                
                '''if self.print_pdes == True:
                                                                    print("Predicted equation is " + print_pde(xi_new,self.description,ut=data_description[i]+'_{t}'))
                                                
                                                                xi_list_no_prune[i].append(xi)'''
                xi_list[i].append(xi_new)
        self.xi_list = xi_list 

        '''        #save results?
                                        if self.save_xi == True:
                                            np.savez(self.write_dir[i],
                                            xi_list = xi_list[i],
                                            xi_list_no_prune=xi_list_no_prune[i],
                                            description=self.description) '''

    def train_val_ODEFind_OOS(self,div_factor):
        
        #create list of final results
        xi_list = [[] for d in self.data_dir]
        xi_list_no_prune = [[] for d in self.data_dir]
        data_description = [chr(117+i) for i in np.arange(len(self.data_dir))]
        
        #load in data source
        mat = [np.load(d+'.npy',allow_pickle=True, encoding='latin1').item() for d in self.data_dir]
        
        T_full = mat[0]['variables'][0]
        U_full = mat[0]['variables'][1][:,0]

        N = len(T_full)
        T = T_full[:N//div_factor]
        U = mat[0]['variables'][1][:N//div_factor,0]
        Ut = [mat[0]['variables'][2][:N//div_factor]]

        
        Theta = np.zeros((len(T),self.deg+1))
        for i in np.arange(self.deg + 1):
            Theta[:,i] = U**i
        description = ['C^'+str(i) for i in np.arange(self.deg+1)]
        description[0] = '1'

        #remove the column of 1's from Theta, desctipion 
        Theta = np.delete(Theta,description.index("1"),axis=1)
        description.remove("1")
        
        
        self.description = description
        self.y = U_full
        
        
        t = np.unique(T)
        x = np.array(1,ndmin=1)
        self.t = T_full

        print("Running " + str(self.reals) + " SinDy Simulations to determine learned equation")
        print("library is " + str(description))

        

        #now perform the training "reals" times
        for r in np.arange(self.reals):
            for i,d in enumerate(self.data_dir):
                
                utTrain,thetaTrain,ptrain,utVal,thetaVal,pval,_,_,_ = data_shuf(Ut[i],Theta,self.shuf_method,self.trainPerc,self.valPerc,len(x),len(t))
                
                xi, hparams, val_score,num_nonzero_u = run_PDE_Find_train_val(thetaTrain,
                                            utTrain,
                                            thetaVal,
                                            utVal,
                                            self.algo_name,
                                            self.description,
                                            lambda_lb=-6, #-4 , -8
                                            lambda_ub=-2, #-1 , -4
                                            deriv_list=self.deriv_index[i])

                val_score_0 = np.min(val_score)

                #perform pruning
                if len(xi[xi!=0]) > 1:
                    xi_new, val_score_new = PDE_FIND_prune_lstsq(xi,
                                                                 utTrain,
                                                                 utVal,
                                                                 thetaTrain,
                                                                 thetaVal,
                                                                 self.description,
                                                                 val_score_0,
                                                                 self.prune_level)
                else:
                    #don't prune if xi only have one nonzero vector
                    xi_new = xi
                
                if self.print_pdes == True:
                    print("Predicted equation is " + print_pde(xi_new,self.description,ut=data_description[i]+'_{t}'))

                xi_list[i].append(xi_new)
        self.xi_list = xi_list 

        '''xi_list_no_prune[i].append(xi)
                                                        xi_list[i].append(xi_new)
                                                        
                                                        #save results?
                                                        if self.save_xi == True:
                                                            np.savez(self.write_dir[i],
                                                            xi_list = xi_list[i],
                                                            xi_list_no_prune=xi_list_no_prune[i],
                                                            description=self.description) '''

    def train_val_ODEFind_SIR(self):
        
        #
        data_description = [chr(117+i) for i in np.arange(len(self.data_dir))]
        
        #load in data source
        mat = [np.load(d+'.npy',allow_pickle=True, encoding='latin1').item() for d in self.data_dir]
        
        
        T = mat[0]['variables'][:,0]
        S = mat[0]['variables'][:,1][:,np.newaxis]
        St = mat[0]['variables'][:,2][:,np.newaxis]
        I = mat[0]['variables'][:,3][:,np.newaxis]
        It = mat[0]['variables'][:,4][:,np.newaxis]

        #pdb.set_trace()

        Ut = np.hstack((St,It))
        
        Theta = np.hstack([S,S**2,I,I**2,I*S])#R,R**2, S*I, S*R, I*R ])
        description = ['S','S^2','I','I^2','IS']#,'R','R^2','SI','SR','IR']
        #Theta = np.hstack([I,S*I,I**2,(S**2)*I,S*(I**2)])
        #description = ["I","SI","I^2","S^2I","SI^2"]

        t = np.unique(T)
        x = np.array(1,ndmin=1)

        self.description = description
        self.S = S
        self.St = St
        self.I = I
        self.It = It
        self.t = t
        self.Theta = Theta
        self.Ut = Ut

        #create list of final results
        xi_list = [[] for d in np.arange(Ut.shape[1])]
        xi_list_no_prune = [[] for d in np.arange(Ut.shape[1])]
        
        
        print("Running " + str(self.reals) + " SinDy Simulations to determine learned equation")
        print("library is " + str(description))

        #now perform the training "reals" times
        for r in np.arange(self.reals):
           
            utTrain,thetaTrain,ptrain,utVal,thetaVal,pval,_,_,_ = data_shuf(Ut,Theta,self.shuf_method,self.trainPerc,self.valPerc,len(x),len(t))

            for i,d in enumerate(np.arange(Ut.shape[1])):
                
                xi, hparams, val_score,num_nonzero_u = run_PDE_Find_train_val(thetaTrain,
                                            utTrain[:,i][:,np.newaxis],
                                            thetaVal,
                                            utVal[:,i][:,np.newaxis],
                                            self.algo_name,
                                            self.description,
                                            lambda_lb=-6, #-4 , -8
                                            lambda_ub=-2, #-1 , -4
                                            deriv_list=self.deriv_index[0])


                val_score_0 = np.min(val_score)

                #perform pruning
                if len(xi[xi!=0]) > 1:
                    xi_new, val_score_new = PDE_FIND_prune_lstsq(xi,utTrain[:,i],utVal[:,i],thetaTrain,thetaVal,self.description,val_score_0,self.prune_level)
                else:
                    #don't prune if xi only have one nonzero vector
                    xi_new = xi
                
                if self.print_pdes == True:
                    print("Predicted equation is " + print_pde(xi_new,self.description,ut=data_description[i]+'_{t}'))

                xi_list_no_prune[i].append(xi)
                xi_list[i].append(xi_new)
        self.xi_list = xi_list
        '''#save results?
                                                if self.save_xi == True:
                                                    np.savez(self.write_dir[0],
                                                    xi_list = xi_list,
                                                    xi_list_no_prune=xi_list_no_prune,
                                                    description=self.description) '''

    '''def logistic_compare(self):
                     
                     
                     #create list of final results
                     xi_list = [[] for d in self.data_dir]
                     xi_list_no_prune = [[] for d in self.data_dir]
                     data_description = [chr(117+i) for i in np.arange(len(self.data_dir))]
                     
                     #load in data source
                     mat = [np.load(d+'.npy',allow_pickle=True, encoding='latin1').item() for d in self.data_dir]
                     
                     
                     T = mat[0]['variables'][0]
                     U = mat[0]['variables'][1][:,0]
                     Ut = [mat[0]['variables'][2]]
                     F = mat[0]['F'][:,0]
                    
                     
                     #Theta = np.zeros((len(T),3))
                     #Theta[:,0] = U
                     #Theta[:,1] = U*(1-U)
                     #Theta[:,2] = U*(1-F*U)
                     #description = ['u','u(1-u)','u(1-Fu)']
                     
                     Theta = np.zeros((len(T),6))
                     #Theta[:,0] = np.ones(U.shape)
                     Theta[:,0] = U
                     Theta[:,1] = U**2
                     Theta[:,2] = U**3
                     Theta[:,3] = F*U
                     Theta[:,4] = F*U**2
                     Theta[:,5] = F*U**3
                     
                     #Theta[:,6] = U**3
                     #Theta[:,7] = U**2*F
                     #Theta[:,8] = U*F**2
                     #Theta[:,9] = F**3
                     
                     description = ['C','C^2','C^3','FC','FC^2','FC^3']#,'C^3','C^2F','CF^2','F^3']
             
                     self.description = description
                     
                     t = np.unique(T)
                     x = np.array(1,ndmin=1)
                     
                     #now perform the training "reals" times
                     for r in np.arange(self.reals):
                         for i,d in enumerate(self.data_dir):
                             
                             
                             #shuffle ut values into train-val data
                             if self.shuf_method == 'neighbors_1d':
                                 utTrain,thetaTrain,ptrain,utVal,thetaVal,pval = data_shuf_1d(Ut[i],Theta,self.trainPerc,self.valPerc,X,T,N_neighbors = 9)
                             elif self.shuf_method == 'neighbors_2d':
                                 utTrain,thetaTrain,ptrain,utVal,thetaVal,pval = data_shuf_2d(Ut[i],Theta,self.trainPerc,self.valPerc,X,Y,T,N_neighbors = 27)
                             else:    
                                 utTrain,thetaTrain,ptrain,utVal,thetaVal,pval,utTest,thetaTest,ptest = data_shuf(Ut[i],Theta,self.shuf_method,self.trainPerc,self.valPerc,len(x),len(t))
                             
                             
                             xi, hparams, val_score,num_nonzero_u = run_PDE_Find_train_val(thetaTrain,utTrain,thetaVal,utVal,self.algo_name,self.description,lambda_lb=-3,lambda_ub=-1,deriv_list=self.deriv_index[i])
                             
             
                             
                             
             
                             
                             ##prune on test data
                             test_score_0 = run_PDE_Find_Test(thetaTest,utTest,xi)
             
                             #perform pruning
                             if len(xi[xi!=0]) > 1: 
                                 xi_new, test_score_new = PDE_FIND_prune_lstsq(xi,utTrain,utTest,thetaTrain,thetaTest,self.description,test_score_0,self.prune_level)
                             else:
                                 xi_new = xi
                             
             
                             #print print_pde(xi_new,description)
                             #prune on validation data
                             #val_score_0 = np.min(val_score)
                             #perform pruning
                             #if len(xi[xi!=0]) > 1:
                             #    xi_new, val_score_new = PDE_FIND_prune_lstsq(xi,utTrain,utVal,thetaTrain,thetaVal,self.description,val_score_0,self.prune_level)
                             #else:
                             #    #don't prune if xi only have one nonzero vector
                             #    xi_new = xi
                             
                             if self.print_pdes == True:
                                 print("Predicted equation is " + print_pde(xi_new,self.description,ut=data_description[i]+'_{t}'))
             
                             xi_list_no_prune[i].append(xi)
                             xi_list[i].append(xi_new)
                             
                             plt.figure()
                             plt.plot(F)
                             plt.plot(ptrain,np.ones(ptrain.shape),'.')
                             plt.title('$'+print_pde_table(xi_new,description)+'$')
                             plt.savefig('learned_eqn'+str(r)+'.png',dvips=500)
                             #save results?
                             if self.save_xi == True:
                                 np.savez(self.write_dir[i] + '_largelib_',
                                 xi_list = xi_list[i],
                                 xi_list_no_prune=xi_list_no_prune[i],
                                 description=self.description)''' 
    

    def logistic_compare(self):
        
        '''
        Determine if the logistic or 2-point distn better describes data
        '''

        #create list of final results
        xi_list = [[] for d in self.data_dir]
        xi_list_no_prune = [[] for d in self.data_dir]
        data_description = [chr(117+i) for i in np.arange(len(self.data_dir))]
        
        #load in data source
        mat = [np.load(d+'.npy',allow_pickle=True, encoding='latin1').item() for d in self.data_dir]
        
        T = mat[0]['variables'][0]
        U = mat[0]['variables'][1][:,0]
        Ut = [mat[0]['variables'][2]]
        F = mat[0]['F'][:,0]
        

        Theta1 = np.zeros((len(T),2))
        Theta2 = np.zeros((len(T),2))
        Theta1[:,0] = U
        Theta1[:,1] = U*(1-U)

        Theta2[:,0] = U
        Theta2[:,1] = U*(1-F*U)

        t = np.unique(T)
        x = np.array(1,ndmin=1)

        model1_count = 0
        model2_count = 0

        xi_list1 = []
        xi_list2 = []
        
        self.description1 = ['C','C(1-C)']
        self.description2 = ['C','C(1-FC)']

        for r in np.arange(self.reals):
            utTrain,theta1Train,ptrain,utTest,theta1Test,ptest,_,_,_ = data_shuf(Ut[0],Theta1,self.shuf_method,self.trainPerc,self.valPerc,len(x),len(t))
            theta2Train = Theta2[ptrain,:]
            theta2Val = Theta2[ptest,:]
            theta2Test = Theta2[ptest,:]


            xi1 = np.linalg.lstsq(theta1Train,utTrain,rcond=-1)[0]            
            xi2 = np.linalg.lstsq(theta2Train,utTrain,rcond=-1)[0]            

            testing_err1 = np.linalg.norm(utTest-np.matmul(theta1Test,xi1))
            testing_err2 = np.linalg.norm(utTest-np.matmul(theta2Test,xi2))

            if testing_err1 < testing_err2:
                xi_list1.append(xi1)
            else:
                xi_list2.append(xi2)
                

        if len(xi_list1) >= len(xi_list2):
            self.xi_list = xi_list1
            self.description = self.description1
            self.votes = len(xi_list1)
        elif len(xi_list2) > len(xi_list1):
            self.xi_list = xi_list2
            self.description = self.description2
            self.votes = len(xi_list2)

        '''#save results?
                                if self.save_xi == True:
                                    np.savez(self.write_dir[i],
                                    xi_list = self.xi_list,
                                    description=self.description,
                                    votes = self.votes,
                                    reals = self.reals) '''

    '''def identifiability_check(self):
                    
                    
                    #Determine if the terms are correlated or not
                    
            
                    mat = [np.load(d+'.npy',allow_pickle=True, encoding='latin1').item() for d in self.data_dir]
                    
                    U = mat[0]['variables'][1][:,0]
                    Ut = [mat[0]['variables'][2]]
                    F = mat[0]['F'][:,0]
                    
                    
                    Theta = np.zeros((len(U),3))
                    Theta1 = np.zeros((len(U),2))
                    Theta2 = np.zeros((len(U),2))
                    
                    Theta[:,0] = U
                    Theta[:,1] = U*(1-U)
                    Theta[:,2] = U*(1-F*U)
            
                    Theta1[:,0] = U
                    Theta1[:,1] = U*(1-U)
            
                    Theta2[:,0] = U
                    Theta2[:,1] = U*(1-F*U)
            
                    moment = np.matmul(Theta.T,Theta)
                    moment1 = np.matmul(Theta1.T,Theta1)
                    moment2 = np.matmul(Theta2.T,Theta2)
                    
                    rcond1 = np.sqrt(np.linalg.cond(moment))
                    rcond2 = np.sqrt(np.linalg.cond(moment1))
                    rcond3 = np.sqrt(np.linalg.cond(moment2))
            
                    data_xi = {}
                    data_xi['rcond1'] = rcond1
                    data_xi['rcond2'] = rcond2
                    data_xi['rcond3'] = rcond3  
                    
                    np.save(self.write_dir[0]+"_conditioning",data_xi)'''

        
        
    def list_common_eqns(self):

        data_description = self.description

        xi_list = self.xi_list
        
        xi_vote = [[] for d in np.arange(len(xi_list))]
        xi_vote_params = [[] for d in np.arange(len(xi_list))]
        #xi_vote_params_SD = [[] for d in self.data_dir]

        #self.write_dir[0] = self.write_dir[0] + "_largelib_"

        for i in np.arange(len(xi_list)):
            
            xi_vote_tmp = []

            for j in range(len(xi_list[i])):
                xi_vote_tmp.append(trans_rev((np.abs(xi_list[i][j]) > 1e-4)*1)[0])

            xi_vote_tmp = Counter(xi_vote_tmp).most_common(self.num_eqns)
            xi_vote[i] = [x[0] for x in xi_vote_tmp]
            
            #help with bookkeeping for obtaining param estimates
            matrix_vote_initialized = [False for j in np.arange(self.num_eqns)]
            A = ["" for j in np.arange(self.num_eqns)]


            #loop through xi estimates
            for j in np.arange(len(xi_list[i])):
                xi_full = xi_list[i][j]
                #find if this xi estimate matches one of our top votes
                match =  trans_rev(np.abs(xi_full) > 1e-4 )*1 == xi_vote[i]
                #if so, add to list in the entry corresponding to that vote
                if np.any(match):
                    if not matrix_vote_initialized[np.where(match)[0][0]]:
                        A[np.where(match)[0][0]] = xi_full
                        matrix_vote_initialized[np.where(match)[0][0]] = True
                    else:
                        A[np.where(match)[0][0]] = np.hstack((A[np.where(match)[0][0]],xi_full))
             
            #save params of mean parameter estimates for each equation
            if len(xi_vote_params[i]) == 0:
                xi_vote_params[i] = [np.mean(A[k],axis=1) for k in np.arange(len(xi_vote[i]))]
            else:
                xi_vote_params[i].append([np.mean(A[k],axis=1) for k in np.arange(len(xi_vote[i]))])
            '''#save SD of mean parameter estimates for each equation
            if len(xi_vote_params_SD[i]) == 0:
                xi_vote_params_SD[i] = [np.std(A[k],axis=1) for k in np.arange(len(xi_vote[i]))]
            else:
                xi_vote_params_SD[i].append([np.std(A[k],axis=1) for k in np.arange(len(xi_vote[i]))])'''
            
        print("Top Learned equation is:")
        if len(xi_vote_params)==1:
            print(print_pde(xi_vote_params[0][0][:,np.newaxis],data_description,ut="dC/dt"))
            self.inferred_xi = xi_vote_params[0]
        if len(xi_vote_params)==2:
            print(print_pde(xi_vote_params[0][0][:,np.newaxis],data_description,ut="dS/dt"))
            print(print_pde(xi_vote_params[1][0][:,np.newaxis],data_description,ut="dI/dt"))
            self.inferred_xi = [xi_vote_params[0],xi_vote_params[1]]



        
        

    def list_common_eqns_SIR(self):


        #data = np.load(self.write_dir[0]+".npz",allow_pickle=True, encoding='latin1')

        xi_list = self.xi_list

        xi_vote = [[] for d in np.arange(len(xi_list))]
        xi_vote_params = [[] for d in np.arange(len(xi_list))]
        xi_vote_params_SD = [[] for d in np.arange(len(xi_vote))]

        pdb.set_trace()

        for i,xi_list in enumerate(data['xi_list']):
                
            xi_vote_tmp = []

            for xi in xi_list:
                xi_vote_tmp.append(trans_rev((np.abs(xi) > 1e-4)*1)[0])

            xi_vote_tmp = Counter(xi_vote_tmp).most_common(self.num_eqns)
            xi_vote[i] = [x[0] for x in xi_vote_tmp]
            
            #help with bookkeeping for obtaining param estimates
            matrix_vote_initialized = [False for j in np.arange(self.num_eqns)]
            A = ["" for j in np.arange(self.num_eqns)]

            #loop through xi estimates
            for xi in xi_list:
                xi_full = xi
                #find if this xi estimate matches one of our top votes
                match =  trans_rev(np.abs(xi_full) > 1e-4 )*1 == xi_vote[i]
                #if so, add to list in the entry corresponding to that vote
                if np.any(match):
                    if not matrix_vote_initialized[np.where(match)[0][0]]:
                        A[np.where(match)[0][0]] = xi_full
                        matrix_vote_initialized[np.where(match)[0][0]] = True
                    else:
                        A[np.where(match)[0][0]] = np.hstack((A[np.where(match)[0][0]],xi_full))

            
            #save params of mean parameter estimates for each equation
            if len(xi_vote_params[i]) == 0:
                xi_vote_params[i] = [np.mean(A[k],axis=1) for k in np.arange(len(xi_vote[i]))]
            else:
                xi_vote_params[i].append([np.mean(A[k],axis=1) for k in np.arange(len(xi_vote[i]))])
            #save SD of mean parameter estimates for each equation
            if len(xi_vote_params_SD[i]) == 0:
                xi_vote_params_SD[i] = [np.std(A[k],axis=1) for k in np.arange(len(xi_vote[i]))]
            else:
                xi_vote_params_SD[i].append([np.std(A[k],axis=1) for k in np.arange(len(xi_vote[i]))])
            
        
        
        print("Top Learned equations are:")
        print(print_pde(xi_vote_params[0][0][:,np.newaxis],data['description'],ut='S_t'))
        print(print_pde(xi_vote_params[1][0][:,np.newaxis],data['description'],ut='I_t'))

        self.inferred_xi = xi_vote_params

        if self.save_learned_xi:
            for i,d in enumerate(self.data_dir):
                #save vectors of most common equations:
                data_xi = {}
                data_xi['xi_vectors'] = xi_vote_params[i]
                data_xi['xi_SD_vectors'] = xi_vote_params_SD[i]
                data_xi['description'] = data['description']
                
                try:
                    data_xi['votes'] = data['votes']
                    data_xi['reals'] = data['reals']
                except:
                    pass
                np.save(self.write_dir[i]+"_xi_results",data_xi)
        
    def simulate_learned_eqns_compare(self):

        ###
        ### Simulate learned equations (PDEs)
        ### starting with IC from first time points
        ### and determine which can best predict
        ### the final timepoint
        ###
        ### currently assuming data has only 1 compartment,
        ### and data is 1d in space

        
        #load in data source
        mat = [np.load('data/'+d+'.npy',allow_pickle=True, encoding='latin1').item() for d in self.data_file][0]
        mat_sf = [np.load(d+'.npy',allow_pickle=True, encoding='latin1').item() for d in self.data_dir][0]
        
        mat_sf_vars = mat_sf['variable_names']

        if len(np.squeeze(mat['inputs']).shape) == 2:
            solve_str = 'pde'
            #PDE
            #variables
            X = mat['inputs'][:,0]
            T = mat['inputs'][:,1]

            x = np.unique(X)
            t = np.unique(T)

            U = mat['outputs']
            U_sf = mat_sf['variables'][mat_sf_vars.index('u')]
            N = len(U)

            U = U.reshape(len(x),len(t))
            U_sf = U_sf.reshape(len(x),len(t))

            m = mat['m']
            

        elif len(np.squeeze(mat['inputs']).shape) == 1:
            solve_str = 'ode'
            #PDE
            #variables
            T = mat['inputs'][:,0]

            t = np.unique(T)

            U = mat['outputs']
            N = len(U)
            
        
        #concatenate first, last timepoints:
        t_sim = t

        if solve_str == 'pde':
            #ensure window odd
            #window = len(x)//2
            #if window % 2 == 0: window+=1
            
            IC = U_sf[:,0]#savgol_filter(U[:,0],11,np.min((5,window-1)))

            RHS = learned_RHS

        elif solve_str == 'ode':
            IC = U[0]

            RHS = learned_RHS_ODE

        #load in learned equations
        le_mat = np.load(self.write_dir[0]+"_xi_results.npy",allow_pickle=True, encoding='latin1').item()
        description = le_mat['description'].tolist()
        xi_list = le_mat['xi_vectors']


        
        y_sim = []
        AIC_learned_eqns = []
        for xi in xi_list:
            #print print_pde(xi,description)
            if solve_str == 'pde':
                y_tmp = PDE_sim(xi,RHS,x,t_sim,IC,description=description)
                y_tmp = y_tmp[:,1:]
                y_sim.append(y_tmp) #ignore time 0
                RSS = np.linalg.norm(y_tmp - U[:,1:])**2
            
            elif solve_str == 'ode':
                y_tmp = ODE_sim(xi,RHS,t_sim,IC,description=description)
                y_tmp = y_tmp[1:]
                y_sim.append(y_tmp) #ignore time 0
                RSS = np.linalg.norm(y_tmp - np.squeeze(U[1:]))**2
            
            k = sum(xi!=0)
            AIC_learned_eqns.append(N*np.log(RSS/N)+2*(k+1))

        #top eqn has lowest AIC
        learned_eqn_index = np.argmin(AIC_learned_eqns)
        y_learned = y_sim[learned_eqn_index]
        AIC_learned_eqns = AIC_learned_eqns[learned_eqn_index]

        data = {}
        data['final_xi'] = xi_list[learned_eqn_index]
        data['learned_AIC'] = AIC_learned_eqns
        data['description'] = description
        data['u_learned'] = y_learned

        np.save(self.write_dir[0]+"_xi_results_final",data)

    def check_sf_results(self):
        #load in true data
        data_true = [np.load(self.true_data_dir[i]+".npy",allow_pickle=True, encoding='latin1').item() for i in np.arange(len(self.true_data_dir))]
        U = [data_true[i]['outputs'].reshape(data_true[i]['shape']) for i in np.arange(len(self.true_data_dir))]
        #load in dep variables
        x = np.unique(data_true[0]['inputs'][:,0])
        t = np.unique(data_true[0]['inputs'][:,1])
        
        data_description = [chr(117+i) for i in np.arange(len(self.data_dir))]
        
        #load in computed deriv values
        data_nn = [np.load(self.data_dir[i]+".npy",allow_pickle=True, encoding='latin1').item() for i in np.arange(len(self.data_dir))]
        U_nn = [data_nn[i]['variables'][2].reshape(data_true[i]['shape']) for i in np.arange(len(self.data_dir))]
        U_x_nn = [data_nn[i]['variables'][3].reshape(data_true[i]['shape']) for i in np.arange(len(self.data_dir))]
        U_t_nn = [data_nn[i]['variables'][4].reshape(data_true[i]['shape']) for i in np.arange(len(self.data_dir))]
        U_xx_nn = [data_nn[i]['variables'][5].reshape(data_true[i]['shape']) for i in np.arange(len(self.data_dir))]


        '''fig,axs = plt.subplots(20,2,figsize=(10,20))
        for i_count, j in enumerate(np.arange(0,100,20)):
            axs[i_count,0].plot(x,U[:,j],'g.',label='Noisy values')
            axs[i_count,0].plot(x,U0[:,j],'b-',label='clean values')
            axs[i_count,0].plot(x,u_nn[:,j],'r--',label='nn predicted value')

            axs[i_count,1].plot(x,V[:,j],'g.')
            axs[i_count,1].plot(x,V0[:,j],'b-')
            axs[i_count,1].plot(x,v_nn[:,j],'r--')

            axs[i_count,0].set_xlabel('x')
            axs[i_count,1].set_xlabel('x')
            axs[i_count,0].set_title('u(x,t), t = '+str(t[j]))
            axs[i_count,1].set_title('v(x,t), t = '+str(t[j]))
            fig.suptitle('NN predictions for realization '+str(i))

            if j == 0:
                fig.legend()'''

        for j in np.linspace(0,len(t)-1,5):
            fig,axs = plt.subplots(1,len(self.data_dir),figsize=(8,5),squeeze=False)
            [axs[0,i].plot(x,U[i][:,np.int(j)],'r.',label='Noisy values') for i in np.arange(len(self.data_dir))]
            [axs[0,i].plot(x,U_nn[i][:,np.int(j)],'b-',label='nn predicted values') for i in np.arange(len(self.data_dir))]
            
            [axs[0,i].set_xlabel('x') for i in np.arange(len(self.data_dir))]
            [axs[0,i].set_title(data_description[i]+'(x,t), t = '+str(t[np.int(j)])) for i in np.arange(len(self.data_dir))]
            fig.suptitle('NN predictions for realization '+self.data_dir[0])
            axs[0,i].legend()


        return U, U_nn
 
    def check_sf_results_gif(self):
        #load in true data
        data_true = [np.load(self.true_data_dir[i]+".npy",allow_pickle=True, encoding='latin1').item() for i in np.arange(len(self.true_data_dir))]
        U = [data_true[i]['outputs'].reshape(data_true[i]['shape']) for i in np.arange(len(self.true_data_dir))]
        #load in dep variables
        x = np.unique(data_true[0]['inputs'][:,0])
        t = np.unique(data_true[0]['inputs'][:,1])
        
        data_description = [chr(117+i) for i in np.arange(len(self.data_dir))]
        
        #load in computed deriv values
        data_nn = [np.load(self.data_dir[i]+".npy",allow_pickle=True, encoding='latin1').item() for i in np.arange(len(self.data_dir))]
        U_nn = [data_nn[i]['variables'][2].reshape(data_true[i]['shape']) for i in np.arange(len(self.data_dir))]
        U_x_nn = [data_nn[i]['variables'][3].reshape(data_true[i]['shape']) for i in np.arange(len(self.data_dir))]
        U_t_nn = [data_nn[i]['variables'][4].reshape(data_true[i]['shape']) for i in np.arange(len(self.data_dir))]
        U_xx_nn = [data_nn[i]['variables'][5].reshape(data_true[i]['shape']) for i in np.arange(len(self.data_dir))]

        
        images = []
        fig,axs = plt.subplots(1,len(self.data_dir),figsize=(8,5),squeeze=False)
        for j in np.linspace(0,len(t)-1,50,dtype=int):
            [axs[0,i].cla() for i in np.arange(len(self.data_dir))]
            [axs[0,i].plot(x,U[i][:,np.int(j)],'r.',label='Noisy values') for i in np.arange(len(self.data_dir))]
            [axs[0,i].plot(x,U_nn[i][:,np.int(j)],'b-',label='nn predicted values') for i in np.arange(len(self.data_dir))]
            
            [axs[0,i].set_xlabel('x') for i in np.arange(len(self.data_dir))]
            [axs[0,i].set_title(data_description[i]+'(x,t), t = '+str(t[np.int(j)])) for i in np.arange(len(self.data_dir))]
            fig.suptitle('NN predictions for realization '+self.data_dir[0])
            axs[0,i].legend()
            
            plt.savefig(self.animations +"/gifs/"+ self.data_file[0] + '_' 
                        + str(j) + ".png",dvips=500)
            images.append(imageio.imread(self.animations +"/gifs/"+ self.data_file[0] + '_' 
                        + str(j) + ".png"))
        imageio.mimsave(self.animations +"/"+ self.data_file[0] + ".gif",images)


        return

        
        #plt.savefig("nn_two_subpopns"+str(i)+".jpg",dvips=1000)
                 
