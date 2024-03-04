from .base import BaseFunction, BaseVariable
from typing import Union


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
        creater_list = [self.creator]
        for f in creater_list:
            x = f.input
            x.grad = f.backward(self.grad)
            if x.creator:
                creater_list.append(x.creator)

    def __repr__(self):
        if self.grad is None:
            return f"<Variable(data={self.data:.2f})>"
        else:
            return f"<Variable(data={self.data:.2f}, grad={self.grad:.2f}, grad=True)>"
