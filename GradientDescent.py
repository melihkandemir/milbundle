import numpy as np

class GradientDescent:
    
    
    def __init__(self,learningrate, func, dfunc, Xinit, args, maxiter,outfilename):
        
        self.lrate = learningrate
        self.func = func
        self.dfunc = dfunc
        self.args = args
        self.maxiter = maxiter
        self.Xinit = Xinit
        self.outfilename=outfilename
        
        
    def minimize(self):
        
        X = self.Xinit
        
        tau = 1.
        kappa = 0.5
        
        for ii in range(self.maxiter):
              
            lrate_ii = self.lrate*pow(ii+tau,-kappa)                                
            
            dX = self.dfunc(X,*self.args)
            
            X -= lrate_ii * dX
            
            #loss = self.func(X,*self.args)
            
            #print '    Loss %6i;  Value %.3f\r' % (ii, loss)
            print '    Iter %6i;\r' % (ii)
            
            
            
        return X



            
        