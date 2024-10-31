import os,sys 
sys.path.append(os.pardir)

from dezero.core import Variable
from dezero.core import Func as F
import numpy as np



x = Variable(np.array(2.))


y = F.add(x, F.add(x,x))
y.backward()

print(y.data)
print(y.grad)
print(x.grad)