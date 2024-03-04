from .base import BaseFunction, BaseVariable
from typing import Union
import numpy as np


class Variable(BaseVariable):
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func: Union[BaseFunction, None] = None):
        self.creator = func

    # def backward(self):
    #     # recursion backward
    #     f = self.creator
    #     if f:
    #         x = f.input
    #         x.grad = f.backward(self.grad)
    #         x.backward()

    def backward(self):
        # loop backward
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x = f.input
            y = f.output  # f.output should be self?
            x.grad = f.backward(y.grad)
            if x.creator:
                funcs.append(x.creator)

    def __repr__(self):
        if self.grad is None:
            return f"<Variable(data={self.data:.2f})>"
        else:
            return f"<Variable(data={self.data:.2f}, grad={self.grad:.2f}, grad=True)>"
