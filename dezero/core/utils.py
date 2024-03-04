import numpy as np


class Utils(object):

    @staticmethod
    def as_array(x):
        # if compute is a scalar then convert it to array
        if np.isscalar(x):
            x = np.array(x)
        return x
