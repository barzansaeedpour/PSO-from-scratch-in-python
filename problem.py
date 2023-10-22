import numpy as np

# The problem is defined in this file

def sphere(x):
    z = np.sum(np.power(x,2))
    return z


def problem(x):

    z = sphere(x)

    return z