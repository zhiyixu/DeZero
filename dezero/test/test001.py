import os  # NOQA: E402
import sys  # NOQA: E402
sys.path.append(os.pardir)  # NOQA: E402
from core import Variable
from core import Square, Exp
import numpy as np


if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()
    x = Variable(np.array(.5))

    a = A(x)
    b = B(a)
    y = C(b)

    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x

    y.grad = np.array(1.)

    y.backward()
    print(a, b, x, y)
