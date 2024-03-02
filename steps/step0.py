

class Variable:
    def __init__(self, data):
        self.data = data 
        
    def __repr__(self):
        return f"<Variable(data={self.data})>"
        
if __name__ == "__main__":
    import numpy as np 
    data = np.array(1.0)
    x = Variable(data)
    print(x.data)
    x.data = np.array(4)
    print(x.data)
    print(x)