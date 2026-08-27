import numpy as np 
import matplotlib.pyplot as plt 

from model import *
from utils import *

X = normalize(X)

w = np.zeros((X.shape[1], 1))
b = 0 

alpha = 0.001
iterations = 1000

w, b , J_history = gradient_descent(X, y, w, b, alpha, iterations)

:
