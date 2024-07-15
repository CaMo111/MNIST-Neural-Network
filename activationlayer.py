from layer_abstractbaseclass import Layer
import numpy as np

class ActiviationLayer(Layer):

    def __init__(self, activation, activation_integral) -> None:
        self.activation = activation
        self.activationprime = activation_integral

    def forward(self, input):
        self.input = input
        output = self.activation(self.input)
        return output
    
    def backward(self, output_gradient, learning_rate):
        #multiply two vectors element-wise
        '''
        The way that this layer works is that it simply applies a singular activation function from past layer
        to curr layer. Thus to get error gradient of input layer we are simply doing dot product from the integral of 
        the nodes activation function to the error loss of the node.
        '''
        return np.multiply(output_gradient, self.activationprime(self.input))
        