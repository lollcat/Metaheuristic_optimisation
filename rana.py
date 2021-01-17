import numpy as np


def rana_func(x):
    return np.sum(x[:-1]*np.cos(np.sqrt(np.abs(x[1:]+x[:-1]+1)))*np.sin(np.sqrt(np.abs(x[1:]-x[:-1]+1))) + \
           (1+x[1:])*np.cos(np.sqrt(np.abs(x[1:]-x[:-1]+1)))*np.sin(np.sqrt(np.abs(x[1:]+x[:-1]+1))))

def Rosenbrock(x):
    return np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

if __name__ == "__main__":
    print(rana_func(np.array([2, 3])))
    print(Rosenbrock(np.array([1, 1])))
