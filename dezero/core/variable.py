from .base import BaseFunction, BaseVariable


class Variable(BaseVariable):
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func: BaseFunction):
        self.creator = func

    def __repr__(self):
        if self.grad is None:
            return f"<Variable(data={self.data:.2f})>"
        else:
            return f"<Variable(data={self.data:.2f}, grad={self.grad:.2f}, grad=True)>"
