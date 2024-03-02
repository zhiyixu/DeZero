

class Variable:
    def __init__(self, data):
        self.data = data 
        self.grad = None
        
    def __repr__(self):
        if self.grad is None:
            return f"<Variable(data={self.data:.2f})>"
        else:
            return f"<Variable(data={self.data:.2f}, grad={self.grad:.2f}, grad=True)>"
        
if __name__ == "__main__":
    import numpy as np 
    data = np.array(1.0)
    x = Variable(data)
    print(x.data)
    x.data = np.array(4)
    print(x.data)
    print(x)