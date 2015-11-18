from melpy.core.Classifier import Classifier
from melpy.validation.ClassPrediction import ClassPrediction
import numpy as np
from scipy.stats import norm
import sklearn.cluster
from copy import deepcopy
from melpy.cyutils.rbfgrads import *
import scipy
from melpy.optimization.StochasticGradientDescent import StochasticGradientDescent
from os import listdir
from os.path import isfile, join
import scipy.io
import sys

class BayesianLogisticRegressionMIL(Classifier):
    
            
    def __init__(self, max_iter = 50,lrate=0.0001,verbose=0,learnhyper=1,islearnthres=1,normalize=1,outfilename=sys.stdout):
        Classifier.__init__(self,'Bayes-LogReg-MIL')
               
        self.max_iter = max_iter             
        self.thres = 0.5
        self.lrate = lrate
        self.verbose = verbose
        self.learnhyper = learnhyper
        self.islearnthres=islearnthres
        self.outfilename=outfilename
        self.normalize=normalize
        
    def train(self,dataset,trainbags):

        self.bagnames = dataset.getBagNames()
                       
        self.state = self.initialize(dataset,trainbags)
        
        gd = StochasticGradientDescent(self.lrate,self.MAP_SGD,self.grad_MAP_SGD,self.state.vectorize(),(dataset,trainbags),self.max_iter,self.outfilename)        
        state_vec = gd.minimize()
        
        self.state = self.state.devectorize(state_vec)   
        
        if self.islearnthres>0:
            self.learnThres(dataset,trainbags)         
            
    def sigmoid(self,x):
      return 1 / (1 + np.exp(-x))        
        
    def MAP_SGD(self, state_vec, bagidx, dataset, trainbags):
        state = self.state.devectorize(state_vec)        
        
        B = trainbags.shape[0] 
        
        mat = scipy.io.loadmat( dataset.path + '/' + self.bagnames[0][trainbags[bagidx]][0] )                       
        X = np.array(mat["DataBag"])            
        X[np.isnan(X)] = 0.
        X[np.isinf(X)] = 0.
        Tbb = mat["label"].ravel()
        Nb = X.shape[0]        
        
        if self.normalize == 1 :
            X = (X - np.tile(self.data_mean,[Nb,1])) / np.tile(self.data_std,[Nb,1])        
            
        pbag = 1.-np.prod(1.-self.sigmoid(X.dot(state.w)))
        
        if pbag==1.0:
            pbag=0.95

        val  = -0.5*(state.w*state.alpha).dot(state.w)
        
        val += B*(Tbb*np.log(pbag) + (1.-Tbb)*np.log(1.-pbag))
        
        return -val        
        
    def grad_MAP_SGD(self, state_vec, bagidx, dataset, trainbags):
        
        state = self.state.devectorize(state_vec)
        dstate = self.state.create_zero_state() # derivative state
        
        B = trainbags.shape[0] 
        
        mat = scipy.io.loadmat( dataset.path + '/' + self.bagnames[0][trainbags[bagidx]][0] )                       
        X = np.array(mat["DataBag"])            
        X[np.isnan(X)] = 0.
        X[np.isinf(X)] = 0.
        Tbb = mat["label"].ravel()
        Nb = X.shape[0]     
        D = X.shape[1]
        
        if self.normalize == 1 :     
           
           X = (X - np.tile(self.data_mean,[Nb,1])) / np.tile(self.data_std,[Nb,1])
           
        pbag = 1.-np.prod(1.-self.sigmoid(X.dot(state.w)))           
        
        beta = (1.-pbag) / pbag
                
        dstate.w = -(state.w*state.alpha)
        
        dstate.w += B*(Tbb*beta-(1.-Tbb))*np.sum(X*np.tile(self.sigmoid(X.dot(state.w)),[D,1]).T,axis=0)
            
        dstate_vec = dstate.vectorize()
        
        return -dstate_vec
        
    def learnThres(self,dataset,bags):
        
        ypred = self.predict(dataset,bags)
        probs = ypred.probabilities
        vals = np.unique(probs)
        
        maxAcc=0.
        maxThr=0.
        for pp in range(vals.shape[0]):
            
            Tpp = (ypred.probabilities>vals[pp])*1
            
            acc=np.mean(self.T==Tpp)
            
            if acc>maxAcc:
                maxAcc=acc
                maxThr=vals[pp]
                
        self.thres = maxThr           
        
    def initialize(self,dataset,trainbags):
        
        B=trainbags.shape[0]
        
        T = np.zeros([B,1]).ravel()
        
        Nrepr_tot=0
                
        for bb in range(B):
                        
            mat = scipy.io.loadmat( dataset.path + '/' + self.bagnames[0][trainbags[bb]][0] )                       
            DataBag = np.array(mat["DataBag"])            
            DataBag[np.isnan(DataBag)] = 0.
            DataBag[np.isinf(DataBag)] = 0.
            Xrepr_bb = np.array(mat["DataRepr"])
            D = DataBag.shape[1]            
            T[bb] = mat["label"]
            
            Nrepr_tot += Xrepr_bb.shape[0]
            
            if bb==0:
                Xmeans = np.zeros([B,DataBag.shape[1]])
                
            Xmeans[bb,:] = np.mean(DataBag,axis=0)
            
            if bb == 0:
                Xrepr = Xrepr_bb
                Trepr = np.ones([Xrepr_bb.shape[0],1]).ravel()*T[bb]
                Xall = DataBag
                Tall =  np.ones([DataBag.shape[0],1]).ravel()*T[bb]
                
            else:
                Xall = np.concatenate((Xall,DataBag),axis=0)       
                Tall = np.concatenate((Tall,np.ones([DataBag.shape[0],1]).ravel()*T[bb]),axis=0)
                
                Xrepr = np.concatenate((Xrepr,Xrepr_bb),axis=0)
                Trepr_bb = np.ones([Xrepr_bb.shape[0],1]).ravel()*T[bb]
                Trepr = np.concatenate((Trepr,Trepr_bb),axis=0)    
                
            if self.verbose > 0:
                  print '    Bag %6i initialized;\r' % (bb)                  
            
        # normalize
        #self.data_mean = np.mean(Xrepr,axis=0)
        #self.data_std = np.std(Xrepr,axis=0)
        #self.data_std[self.data_std==0.] = 1.   
        
        self.data_mean = np.mean(Xall,axis=0)
        self.data_std = np.std(Xall,axis=0)
        self.data_std[self.data_std==0.] = 1.
        
        if self.normalize == 1 :
            Xall = (Xall - np.tile(self.data_mean,[Xall.shape[0],1])) / np.tile(self.data_std,[Xall.shape[0],1])        
        
        self.T = T
        Tall[Tall==0] = -1.
        
        C=Xall.T.dot(Xall)
        Cinv = np.linalg.inv(C+np.identity(D)*0.001)
        w = Cinv.dot(Xall.T).dot(Tall)
        
        
        #w = np.random.random([D,1]).ravel()
        alpha = np.random.random([D,1]).ravel()        
        
        state = State(w,alpha)
        return state        
        
    def predict(self,dataset,testbags):
        
      #Xts_red  = self.pca.transform(Xts)        
      
      B = testbags.shape[0]  
      probabilities = np.zeros([B,1]).ravel()
      
      for bb in range(B):
          
                        
            mat = scipy.io.loadmat( dataset.path + '/' + self.bagnames[0][testbags[bb]][0] )                       
            Xts = np.array(mat["DataBag"])            
            Xts[np.isnan(Xts)] = 0.
            Xts[np.isinf(Xts)] = 0.
            Nb = Xts.shape[0]        
            
            if self.normalize == 1 :            
                Xts = (Xts - np.tile(self.data_mean,[Nb,1])) / np.tile(self.data_std,[Nb,1])
        
            #pbag = 1.-np.prod(1.-self.sigmoid(Xts.dot(self.state.w)))  
            
            pbag = np.max(self.sigmoid(Xts.dot(self.state.w)))
           
            probabilities[bb] = pbag  
            
            if self.verbose > 0:
                  print '    Bag %6i predicted;\r' % (bb)       
            
      classes = (probabilities>self.thres)*1
      class_prediction=ClassPrediction(classes,probabilities)        
      
      return class_prediction       
      
class State:
    
     def __init__(self, w, alpha):
         self.set( w, alpha)
         
     def set(self, w, alpha):         
         self.w = deepcopy(w)
         self.alpha = deepcopy(alpha)               
         
     def create_zero_state(self):         
         
         D = self.w.shape[0]
         
         w = np.zeros([D,1]).ravel()         
         alpha = np.zeros([D,1]).ravel()         
         
         snew = State(w, alpha)
         
         return snew
         
     def vectorize(self):
                
         state_vec = np.concatenate((self.w,self.alpha))
        
         return state_vec
    
     def devectorize(self,state_vec):
                  
         D = self.w.shape[0]
         
         w=state_vec[0:D]
         alpha=state_vec[D:2*D]
         
         snew = State(w,alpha)
         
         return snew
