import os,sys # NOQA: E402
sys.path.append(os.pardir) # NOQA: E402

import unittest
from dezero.core import Variable
from dezero.core import Func as F
from dezero.core import numerical_diff
import numpy as np

class SquareTest(unittest.TestCase):

    def test_square(self):
        x = Variable(np.array(2))
        y = F.square(x)
        z = Variable(np.array(4.))
        self.assertEqual(y.data, z.data)

    def test_backwards(self):
        x = Variable(np.random.rand(1))
        y = F.exp(x)
        y.backward()
        z = numerical_diff(f=F.exp, x=x)
        flg = np.allclose(x.grad,z.data)
        self.assertTrue(flg)

    def test_add(self):
        x = Variable(np.array(3))
        y = Variable(np.array(.8))
        z = F.add(x,y)
        self.assertEqual(z.data, np.array(3.8))


x = Variable(np.array(2.))
y = Variable(np.array(3.))

z = F.add(F.square(x), F.square(y))
z.backward()

print(z.data)
print(x.grad)
print(y.grad)
