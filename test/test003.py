import os,sys 
sys.path.append(os.pardir)

from dezero.core import Variable
from dezero.core import Func as F
import numpy as np



x = Variable(np.array(2.))

a = F.square(x)
y = F.add(F.square(a), F.square(a))

y.backward()

print(y.data)
print(y.grad)
print(x.grad)