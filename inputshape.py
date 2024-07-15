from layer_abstractbaseclass import Layer
import numpy as np

class Reshape(Layer):
    def __init__(self, input, output) -> None:
        self.input = input
        self.output = output

    def forward(self):
        return np.reshape(self.input, self.output)
    
    def backward(self, output_grad, learning_rate):
        return np.reshape(output_grad, self.input)