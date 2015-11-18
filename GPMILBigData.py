from melpy.core.Classifier import Classifier
from melpy.validation.ClassPrediction import ClassPrediction
import numpy as np
from scipy.stats import norm
import sklearn.cluster
import scipy.misc
from copy import deepcopy
from melpy.cyutils.rbfgrads import *
import scipy
from melpy.optimization.GradientDescent import GradientDescent
from os import listdir
from os.path import isfile, join
import scipy.io
import sys
import cv2

class GPMILBigData(Classifier):
    
            
    def __init__(self,kernel, num_inducing = 10, max_iter = 50,lrate=0.0001,islearnthres=1):
        Classifier.__init__(self,'VSGPMIL')
               
        self.kernel=kernel        
        self.P = num_inducing        
        self.max_iter = max_iter
        self.thres = 0.5           
        self.lrate = lrate
        self.islearnthres = islearnthres
        
    def train(self,dataset,trainbags):

        self.bagnames = dataset.getBagNames()
                       
        self.state = self.initialize(dataset,trainbags)
        
                    

        gd = GradientDescent(self.lrate,self.logPosterior,self.grad_logPosterior,self.state.vectorize(),(dataset,trainbags),self.max_iter, sys.stdout)        
        state_vec = gd.minimize()

        
        self.state = self.state.devectorize(state_vec)    
        
        if self.islearnthres==1:
            self.learnThres(dataset,trainbags)
     
        
        
    def logPosterior(self, state_vec, dataset, trainbags):
        state = self.state.devectorize(state_vec)        
        
        B = trainbags.shape[0] 
        
        val = -0.5*state.u.dot(state.Kzz_inv).dot(state.u)
        
        inst_pointer = 0
        
        for bb in range(B):
            mat = scipy.io.loadmat( dataset.path + '/' + self.bagnames[0][trainbags[bb]][0] )                       
            X = np.array(mat["DataBag"])            
            X[np.isnan(X)] = 0.
            X[np.isinf(X)] = 0.
            Tbb = 2.*mat["label"].ravel()-1.
            Nb = X.shape[0]        
            X = (X - np.tile(self.data_mean,[Nb,1])) / np.tile(self.data_std,[Nb,1])
                        
            Kzx = state.kernel.compute(state.Z,X)
            Lambda_inv = np.diag(1. / (state.kernel.compute(X,X).diagonal() - Kzx.T.dot(state.Kzz_inv).dot(Kzx).diagonal()))
            fb = state.f[inst_pointer:inst_pointer+Nb]
            A = Kzx.T.dot(state.Kzz_inv).dot(state.u)
            
            val += -0.5*fb.dot(Lambda_inv).dot(fb)            
            val += -0.5*A.dot(Lambda_inv).dot(A)
            val += fb.dot(Lambda_inv).dot(A)            
            val += -0.5*np.sum( np.log(Lambda_inv.diagonal() ))            
            val += np.log( self.likSoftMax(fb,Tbb) )
#
# TODO: add logdetKzz
            
            inst_pointer += Nb
            
            
        return -val        
        
    def grad_logPosterior(self, state_vec, dataset, trainbags):
        
        state = self.state.devectorize(state_vec)
        dstate = self.state.create_zero_state() # derivative state
        
        B = trainbags.shape[0]
        
        dstate.u = -state.Kzz_inv.dot(state.u)
        
        for bb in range(B):
       
            Tbb = self.T[bb]
            fb = state.f[self.bags==bb]
            
            Abb = self.A[self.bags==bb]
            Bbb_inv = 1./self.B[self.bags==bb]
            
            Bbb_inv_mat = np.tile(Bbb_inv,[self.P,1]).T
            
            dstate.u += -(Abb.T *Bbb_inv_mat.T).dot(Abb).dot(state.u)
            
            dstate.u +=  (fb*Bbb_inv).dot(Abb)

            dstate.f[self.bags==bb] =(Abb*Bbb_inv_mat).dot(state.u) -Bbb_inv*fb + self.grad_likSoftMax(fb,Tbb)
              
                
        dstate_vec = dstate.vectorize()
        
        return -dstate_vec                
        
    def initialize(self,dataset,trainbags):
        
        P = self.P
        Ppos = np.uint32(np.floor(P*0.5))
        B= trainbags.shape[0]
        
        T = np.zeros([B,1]).ravel()
        
        Nrepr_tot = 0
        
        for bb in range(B):
                        
            mat = scipy.io.loadmat( dataset.path + '/' + self.bagnames[0][trainbags[bb]][0] )                       
            DataBag = np.array(mat["DataBag"])            
            DataBag[np.isnan(DataBag)] = 0.
            DataBag[np.isinf(DataBag)] = 0.
            Xrepr_bb = np.array(mat["DataRepr"])

            Nbb = Xrepr_bb.shape[0]            
            Nrepr_tot += Nbb 
            
            #if bb==0:
            #    Xmeans = np.zeros([B,DataBag.shape[1]])
            
            T[bb] = 2.*mat["label"]-1.
            #Xmeans[bb,:] = np.mean(DataBag,axis=0)
            Nb = DataBag.shape[0]            
            
            if bb == 0:
                Xrepr = deepcopy(Xrepr_bb)
                Trepr = np.ones([Nbb,1]).ravel()*T[bb]
            else:
                Xrepr = np.concatenate((Xrepr,Xrepr_bb),axis=0)
                Trepr_bb = np.ones([Nbb,1]).ravel()*T[bb]
                Trepr = np.concatenate((Trepr,Trepr_bb),axis=0)        
                
        self.T = T
            
        # normalize
        self.data_mean = np.mean(Xrepr,axis=0)
        self.data_std = np.std(Xrepr,axis=0)
        self.data_std[self.data_std==0.] = 1.
           
        Xrepr = (Xrepr - np.tile(self.data_mean,[Nrepr_tot,1])) / np.tile(self.data_std,[Nrepr_tot,1])
        
        
        Xrepr = np.float32(Xrepr) 
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        rets, lbls, Z0 = cv2.kmeans(Xrepr[Trepr==-1,:], P-Ppos, criteria, 10, 0)
        rets, lbls, Z1 = cv2.kmeans(Xrepr[Trepr==1,:], Ppos, criteria, 10, 0)
        self.Z = np.concatenate((Z0,Z1))        
                
        Z = np.concatenate((Z0,Z1))
             
                  
        Kzz = self.kernel.compute(Z,Z)
        Kzz_inv = np.linalg.inv(Kzz+np.identity(P)*0.001)
        Kzx = self.kernel.compute(Z,Xrepr)   
        
        u=np.linalg.lstsq(Kzx.T.dot(Kzz_inv),Trepr)[0] 
        
         
        for bb in range(B):
            print "Bag %i preprocessed" % (bb)            
            mat = scipy.io.loadmat( dataset.path + '/' + self.bagnames[0][trainbags[bb]][0] )                       
            Xbb = np.array(mat["DataBag"])            
            Xbb[np.isnan(Xbb)] = 0.
            Xbb[np.isinf(Xbb)] = 0.
            Nb = Xbb.shape[0]
            
            Xbb = (Xbb - np.tile(self.data_mean,[Nb,1])) / np.tile(self.data_std,[Nb,1])
            Kzx = self.kernel.compute(Z,Xbb)    
            Abb = Kzx.T.dot(Kzz_inv)
            fb = Abb.dot(u)
            #fb = np.ones([Nb,1]).ravel()*T[bb]
            
            L_Kzz_inv = np.linalg.cholesky(Kzz_inv)
            Ctemp = Kzx.T.dot(L_Kzz_inv)
            Bbb = np.ones([Nb,1]).ravel()- np.sum(Ctemp*Ctemp,axis=1) 
            
            if bb == 0:
                f  = fb
                A = Abb
                B = Bbb
                bags = np.ones([Nb,1]).ravel()*bb
            else:
                f  = np.concatenate((f,fb ))
                A = np.concatenate((A,Abb))
                B = np.concatenate((B,Bbb))
                bags = np.concatenate((bags,np.ones([Nb,1]).ravel()*bb))
                
        #f = f*0.1
        #u=np.linalg.lstsq(A,f)[0]  
        
        state = State(u,f,Z,Kzz_inv,self.kernel.length_scale,self.kernel.clone())
        
        self.bags = bags;
        self.A = A
        self.B = B
        
        return state
        
    def likSoftMax(self,fb,Tb):
        
        return 1./(1.+np.power(np.exp(scipy.misc.logsumexp(fb)),-Tb))
        
        
    def grad_likSoftMax(self,fb,Tb):
        
        sumExpF = np.exp(scipy.misc.logsumexp(fb))
        
        return Tb*pow(sumExpF,-Tb-1)*np.exp(fb) / (1+np.power(sumExpF,-Tb))
        
    def learnThres(self,dataset,bags):
        
        ypred = self.predict(dataset,bags)
        probs = ypred.probabilities
        vals = np.unique(probs)
        
        maxAcc=0.
        maxThr=0.
        for pp in range(vals.shape[0]):
            
            Tpp = (ypred.probabilities>vals[pp])*1
            Tpp[Tpp==0] = -1.
            
            acc=np.mean(self.T==Tpp)
            
            if acc>maxAcc:
                maxAcc=acc
                maxThr=vals[pp]
                
        self.thres = maxThr
        
    def predict(self,dataset,testbags):
        
        B = testbags.shape[0]  
        probabilities = np.zeros([B,1]).ravel()
        
        for bb in range(B):
            print "Bag %i predicted" % (bb)             
            mat = scipy.io.loadmat( dataset.path + '/' + self.bagnames[0][testbags[bb]][0] )                       
            Xts = np.array(mat["DataBag"])            
            Xts[np.isnan(Xts)] = 0.
            Xts[np.isinf(Xts)] = 0.
            Nb = Xts.shape[0]        
            Xts = (Xts - np.tile(self.data_mean,[Nb,1])) / np.tile(self.data_std,[Nb,1])
        
            Kxz = self.state.kernel.compute(Xts,self.state.Z)                  
            fb = Kxz.dot(self.state.Kzz_inv).dot(self.state.u)     
            #probabilities[bb] = self.likSoftMax(fb,1.0)
            probabilities[bb] = norm.cdf(max(fb))
                 
                 
        classes = (probabilities>self.thres)*1
        #classes = (probabilities>0.5)*1
        class_prediction=ClassPrediction(classes,probabilities)
        
        return class_prediction
      
class State:
    
     def __init__(self, u, f, Z, Kzz_inv, length_scale, kernel):
         self.set(u, f, Z, Kzz_inv, length_scale, kernel)
         
     def set(self,u, f, Z, Kzz_inv, length_scale, kernel):         
         self.u = deepcopy(u)
         self.f = deepcopy(f)
         self.Z = deepcopy(Z)
         self.Kzz_inv = deepcopy(Kzz_inv)
         self.length_scale = kernel.length_scale
         self.kernel=kernel.clone()  
         
     def setZ(self,Znew):         
         self.Z = Znew

         Kzz = self.kernel.compute(Znew,Znew)
         self.Kzz_inv = np.linalg.inv(Kzz+np.identity(Znew.shape[0])*0.001)                 
         
     def create_zero_state(self):         
         
         P = self.Kzz_inv.shape[0]
         D = self.Z.shape[1]
         
         u = np.zeros([P,1]).ravel()         
         f = np.zeros([self.f.shape[0],1]).ravel()
         Z = np.zeros([P,D])         
         Kzz_inv = np.zeros([P,P])         
         snew = State(u,f,Z,Kzz_inv,0.,self.kernel.clone())
         
         return snew
         
     def vectorize(self):
        
         P = self.Kzz_inv.shape[0]
         D = self.Z.shape[1]
                 
         state_vec = np.concatenate((self.u,self.f,np.reshape(self.Z,[P*D,1]).ravel(),np.reshape(self.Kzz_inv,[P*P,1]).ravel(),np.ones([1,1]).ravel()*self.length_scale))
        
         return state_vec
    
     def devectorize(self,state_vec):
                  
         P = self.Kzz_inv.shape[0]
         D = self.Z.shape[1]
         N = self.f.shape[0]
                                     
         u = state_vec[0:P]
         f =state_vec[P:(P+N)]
         Z = np.reshape(state_vec[(P+N):(P+N+P*D)],[P,D])
         length_scale = state_vec[P+N+P*D+P*P]
         kernel = self.kernel.clone()
         kernel.length_scale = length_scale
         
         Kzz = kernel.compute(Z,Z)
         Kzz_inv = np.linalg.inv(Kzz+np.identity(Z.shape[0])*0.001)  
         
         snew = State(u,f,Z,Kzz_inv,length_scale,kernel)
         
         return snew
