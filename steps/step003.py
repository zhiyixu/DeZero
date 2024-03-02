from steps import Variable, Function
import numpy as np 

class Exp(Function):
    
    def _forward(self, x):
        return np.exp(x)