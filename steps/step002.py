from steps import Variable
from steps import Function

class Square(Function):
    
    def _forward(self, x):
        return  x**2
    