from .base import BaseFunction, BaseVariable
from typing import Union
import numpy as np


class Variable(BaseVariable):
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                return TypeError(f"{type(data)} is not support, currently np.ndarray only.")
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func: Union[BaseFunction, None] = None):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):
        # loop backward
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set() 

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            # o is a weakref, thus we take o() instead of o
            gys = [o().grad for o in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad=gx
                else:
                    x.grad = x.grad + gx  # x.grad += gx will cause error 

                if x.creator is not None:
                    add_func(x.creator)

    
    def clean_grad(self):
        self.grad = None
    
    def __repr__(self):
        if self.grad is None:
            return f"<Variable(data={self.data})>"
        else:
            return f"<Variable(data={self.data}, grad={self.grad}, grad=True)>"
