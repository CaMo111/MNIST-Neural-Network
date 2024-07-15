#sigmoid and tanh
from activationlayer import ActiviationLayer
import numpy as np 

class TanH(ActiviationLayer):
    def __init__(self) -> None:
        # could put these into lambda functions as of tutorial
        def calctanh(x):
            return np.tanh(x)
        def calctanh_integral(x):
            return 1 - np.tanh(x)** 2
        
        tanh = calctanh
        tanh_prime = calctanh_integral
        super().__init__(tanh, tanh_prime)

#   loss function calculator
def meansqaureerror(true_output, predicted_output):
    return np.mean(np.power(true_output - predicted_output, 2))

def meansquareerror_integral(true_output, predicted_output):
    return 2 * (predicted_output - true_output) / np.size(true_output)