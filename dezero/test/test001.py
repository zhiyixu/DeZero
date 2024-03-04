import os  # NOQA: E402
import sys  # NOQA: E402
sys.path.append(os.pardir)  # NOQA: E402
from core import Variable
from core import Func
import numpy as np


if __name__ == "__main__":
    x = Variable(np.array(.5))

    a = Func.square(x)
    b = Func.exp(a)
    y = Func.square(b)

    y.grad = np.array(1.)

    y.backward()
    print(a, b, x, y)
