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
    x = Variable(np.array(.5))
    A = Square()
    B = Exp()
    C = Square()
    a = A(x)
    b = B(a)
    c = C(b)
    dy = numerical_diff(f=f_composite,x=c)
    print(dy, dy.data)
    
    y = c 
    y.grad = np.array(1.0) # dy/dc
    b.grad = C.backward(y.grad) # dc/db
    a.grad = B.backward(b.grad) # db/da
    x.grad = A.backward(a.grad) # da/dx
    # ==> dy/dx = dy/dc * dc/db * db/da * da/dx
    print(x, x.grad)