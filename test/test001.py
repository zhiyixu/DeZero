import os,sys 
sys.path.append(os.pardir)
from steps.step003 import Exp
from steps.step002 import Square
from steps.step000 import Variable
from steps import numerical_diff


def f_composite(x:Variable) -> Variable:
    A = Square()
    B = Exp()
    C = Square()
    return A(B(C(x)))
    
if __name__ == "__main__":
    import numpy as np 
    a = np.array(.5)
    a = Variable(a)

    dy = numerical_diff(f=f_composite,x=a)
    print(type(dy), dy.data)