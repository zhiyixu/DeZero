from steps import Variable, Function
import numpy as np 

class Exp(Function):
    
    def _forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self._input.data
        gx = np.exp(x) * gy
        return gx # Variable(gx)