import numpy as np
from abc import abstractmethod
from core import Variable
from .base import BaseVariable, BaseFunction
from .utils import Utils


class Function(BaseFunction):

    def __call__(self, input: BaseVariable) -> BaseVariable:
        if not isinstance(input, (BaseVariable,)):
            raise TypeError(
                f"{type(self).__name__} class require Variable as input, got {type(input)}")
        self.input = input
        x = input.data
        y = self.forward(x)
        self.output = Variable(Utils.as_array(y))
        self.output.set_creator(self)  # sotre the creator
        return self.output

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, gy):
        # this func should be public, loss func will call backward manually
        raise NotImplementedError()


class Square(Function):
    # x: the grad from up stream
    def forward(self, x):
        return x**2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx  # Variable(gx) is better?


class Exp(Function):

    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx  # Variable(gx)


class Func(BaseFunction):

    def square(x: BaseVariable):
        f = Square()
        return f(x)

    def exp(x: BaseVariable):
        f = Exp()
        return f(x)
