import numpy as np


def rana_func(x):
    return np.sum([x[i]*np.cos(np.sqrt(np.abs(x[i+1]+x[i]+1)))*np.sin(np.sqrt(np.abs(x[i+1]-x[i]+1))) +
                   (1+x[i+1])*np.cos(np.sqrt(np.abs(x[i+1]-x[i]+1)))*np.sin(np.sqrt(np.abs(x[i+1]+x[i]+1)))
                   for i in range(0, len(x)-1)])

if __name__ == "__main__":
    rana_func(np.array([2, 3]))
