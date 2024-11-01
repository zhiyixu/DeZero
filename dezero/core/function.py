import numpy as np
from abc import abstractmethod
from .variable import Variable
from .base import BaseVariable, BaseFunction
from .utils import Utils
import weakref

class Function(BaseFunction):

    def __call__(self, *inputs: BaseVariable) -> BaseVariable:
        for in_data in inputs:
            if not isinstance(in_data, (BaseVariable,)):
                raise TypeError(
                    f"{type(self).__name__} class require Variable as input, got {type(in_data)}")
            
        self.inputs = inputs
        xs = [x.data for x in self.inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        self.outputs = [weakref.ref(Variable(Utils.as_array(y))) for y in ys]
        self.generation = max([x.generation for x in self.inputs])
        for o in self.outputs:
            o.set_creator(self)
        return self.outputs[0] if len(self.outputs)==1 else self.outputs

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
        x = self.inputs[0].data  # inputs is a tuple
        gx = 2 * x * gy
        return gx   # Variable(gx) is better?


class Exp(Function):

    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        xs = self.inputs
        gx = tuple([np.exp(x.data) * gy for x in xs])
        return gx[0] if len(gx)==1 else gx  # Variable(gx)

class Add(Function):

    def forward(self, a, b):
        return a+b
    
    def backward(self, gy):
        """
        why return gy twice: in add func, we have two inputs and one output
        when backward, we only call var.backward onece, but we should return both grad
        """
        return gy, gy # only return gy
        

class Func(BaseFunction):
    """
    wapper of funcs above
    """
    def square(x: BaseVariable):
        f = Square()
        return f(x)

    def exp(x: BaseVariable):
        f = Exp()
        return f(x)
    
    def add(x: BaseVariable, y: BaseVariable):
        f = Add()
        return f(x,y)
