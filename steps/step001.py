from abc import ABC, abstractmethod
from steps import Variable


class Function(ABC):

    def __call__(self, input: Variable) -> Variable:
        if not isinstance(input, (Variable,)):
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
