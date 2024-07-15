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
        return np.dot(self.weights, self.input) + self.bias #matrix multiplication of all nodes.
    
    def backward(self, output_gradient, learning_rate):
        '''
        '''
        weights_gradient = np.dot(output_gradient, self.input.T) #see read me calculating weights gradient sect
        input_gradient = np.dot(self.weights.T, output_gradient) # multiplying the weights matrix by the matrix representing the error gradient column matrix
        self.weights -= learning_rate * weights_gradient 
        self.bias -= learning_rate * output_gradient
        return input_gradient