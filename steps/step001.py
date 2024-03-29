from abc import  abstractmethod
from .basetypes import BaseFunction,BaseVariable
from .step000 import Variable

class Function(BaseFunction):

    def __call__(self, input: BaseVariable) -> BaseVariable:
        if not isinstance(input, (BaseVariable,)):
            raise ValueError(
                f"{type(self).__name__} class require Variable as input, got {type(input)}")
        self._input = input
        x = input.data
        y = self._forward(x)
        o = Variable(y)
        o.set_creator(self)
        self._o = o
        return o

    @abstractmethod
    def _forward(self, x):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, gy):
        # this func should be public, loss func will call backward manually
        raise NotImplementedError()
