from steps import Variable
from steps import Function

class Square(Function):
    
    def _forward(self, x):
        return  x**2
    
    def backward(self, gy):
        x = self._input.data
        gx = 2 * x * gy
        return gx # Variable(gx) is better?
    