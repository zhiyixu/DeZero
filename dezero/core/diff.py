from .variable import Variable
from .function import Function


def numerical_diff(f: Function, x: Variable, eps: float = 1e-4) -> Variable:
    x0 = Variable(x.data+eps)
    x1 = Variable(x.data-eps)
    y0 = f(x0)
    y1 = f(x1)
    diff = Variable((y0.data-y1.data) / (2*eps))
    return diff
