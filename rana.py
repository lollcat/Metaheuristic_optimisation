import numpy as np


def rana_func(x):
    return np.sum(x[:-1]*np.cos(np.sqrt(np.abs(x[1:]+x[:-1]+1)))*np.sin(np.sqrt(np.abs(x[1:]-x[:-1]+1))) + \
           (1+x[1:])*np.cos(np.sqrt(np.abs(x[1:]-x[:-1]+1)))*np.sin(np.sqrt(np.abs(x[1:]+x[:-1]+1))))

if __name__ == "__main__":
    rana_func(np.array([2, 3]))
