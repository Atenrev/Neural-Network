import numpy as np

def sigmoid(v):
    return 1 / (1+np.exp(-v))