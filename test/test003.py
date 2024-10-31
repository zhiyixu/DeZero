import os,sys 
sys.path.append(os.pardir)

from dezero.core import Variable
from dezero.core import Func as F
import numpy as np



x = Variable(np.array(2.))
y = Variable(np.array(3.))

z = F.add(F.square(x), F.square(y))
z.backward()

print(z.data)
print(x.grad)
print(y.grad)
