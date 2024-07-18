from layer_abstractbaseclass import Layer
import numpy as np

class DenseLayer(Layer):

    def __init__(self, input_size, output_size):
        '''
        using randn for normal distribution good for our 3 key steps in initalising parameters. eg small, not identical and 
        good variance.
        '''
        self.weights = np.random.randn(output_size, input_size) #calculate random weight inbetween output and input size
        self.bias = np.random.randn(output_size, 1) #calculate random weight 

    def forward(self, input):
        self.input = input 
        #if not change to try and except statement
        try:
            return np.dot(self.weights, self.input) + self.bias
        except:
            return np.dot(self.input, self.weights) + self.bias #matrix multiplication of all nodes.
    
    def backward(self, output_gradient, learning_rate):
        '''
        '''
        try:
            weights_gradient = np.dot(output_gradient, self.input.T)
        except:
            weights_gradient = np.dot(self.input.T, output_gradient) #matrix multiplication of all nodes.
        try:
            input_gradient = np.dot(self.weights.T, output_gradient)
        except:
            input_gradient = np.dot(output_gradient, self.weights.T)
        #update values
        try:
            self.weights = self.weights - (learning_rate * weights_gradient)
        except:
            self.weights = self.weights - (weights_gradient  * learning_rate)
        try:
            self.bias = self.bias - (learning_rate * output_gradient)
        except:
            self.bias = self.bias - (output_gradient * learning_rate)

        return input_gradient