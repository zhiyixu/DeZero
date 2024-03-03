from core import Function

class Variable:
    def __init__(self, data):
        self.data = data 
        self.grad = None
        self.creator = None
        
    def set_creator(self, func:Function):
        self.creator = func

    def __repr__(self):
        if self.grad is None:
            return f"<Variable(data={self.data:.2f})>"
        else:
            return f"<Variable(data={self.data:.2f}, grad={self.grad:.2f}, grad=True)>"