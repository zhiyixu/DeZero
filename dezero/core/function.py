import numpy as np
from abc import abstractmethod
from core import Variable
from .base import BaseVariable, BaseFunction


class Function(BaseFunction):

    def __call__(self, input: BaseVariable) -> BaseVariable:
        if not isinstance(input, (BaseVariable,)):
            raise ValueError(
                f"{type(self).__name__} class require Variable as input, got {type(input)}")
        self._input = input
        x = input.data
        y = self._forward(x)
        o = Variable(y)
        return o

    @abstractmethod
    def _forward(self, x):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, gy):
        # this func should be public, loss func will call backward manually
        raise NotImplementedError()


class Square(Function):

    def _forward(self, x):
        return x**2

    def backward(self, gy):
        x = self._input.data
        gx = 2 * x * gy
        return gx  # Variable(gx) is better?


class Exp(Function):

    def _forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self._input.data
        gx = np.exp(x) * gy
        return gx  # Variable(gx)
