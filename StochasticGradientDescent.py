import numpy as np
from melpy.optimization.GradientDescent import GradientDescent
import sys

class StochasticGradientDescent(GradientDescent):
    
    
    def __init__(self,learningrate, func, dfunc, Xinit, args, maxiter,outfilename):
        
        GradientDescent.__init__(self,learningrate, func, dfunc, Xinit, args, maxiter,outfilename)
        
    def minimize(self):
        
        X = self.Xinit
        
        tau = 1.
        kappa = 0.5
        
        for ii in range(self.maxiter):
            
            indices=np.unique(self.args[1]).astype(np.uint32)
            
            chosen_point = np.random.randint(indices.shape[0])
            
            lrate_ii = self.lrate*pow(ii+tau,-kappa)                                
            
            dX = self.dfunc(X,chosen_point,*self.args)
            
            X -= lrate_ii * dX

            
            if self.outfilename == 'none':
                xxx=0                      
            elif self.outfilename == sys.stdout:
                f = self.outfilename
            else:
                f = open(self.outfilename,'w')
            
            loss = self.func(X,chosen_point,*self.args)
            
            if self.outfilename != 'none': 
                f.write('    Loss %6i;  Value %.3f\r' % (ii, loss))

            if self.outfilename != 'none' and self.outfilename != sys.stdout:            
                f.close()
            
            
            
        return X



            
        