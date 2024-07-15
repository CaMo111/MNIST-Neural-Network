import numpy as np 
import pandas as pd
import denselayer
import activationandlosses

class NeuralNetwork:

    def __init__(self) -> None:
        self.dev_labels, self.dev_data, self.training_labels, self.training_data = self.process_data("mnist_train.csv")

    def process_data(self, filepath_or_buffer: str):
        '''
        number of examples is m
        number of pixels is n 
        Output four items, the dev labels, dev data, training labels and training data 
        '''
        data = pd.read_csv(filepath_or_buffer)
        data = np.array(data)
        m, n = data.shape
        np.random.shuffle(data)
        #for cross checking our data, transpose the matrix so each entry is 
        data_dev = data[0:1000].T
        dev_labels = data_dev[0]
        dev_data = data_dev[1:n]
        data_training = data[1000:m].T
        training_labels = data_training[0]
        training_data = data_training[1:n]
        #normalisation between 0-1 range
        for ith_pixel in dev_data:
            for unique in ith_pixel:
                unique = (1/255) * unique
        for ith_pixel in training_data:
            for unique in ith_pixel:
                unique = (1/255) * unique
        return dev_labels, dev_data, training_labels, training_data
    
    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, image, label, loss=activationandlosses.meansqaureerror, lossprime=activationandlosses.meansquareerror_integral,epochs=59000, learning_rate=0.1):
        for e in range(epochs):
            error = 0
        for image, label in zip(image, label):
            #forward
            output = self.predict(image)
            #calculate loss-entropy
            error += loss(label, output)
            #back propogation
            gradient = lossprime(label, output)
            for layer in reversed(self.layers):
                gradient = layer.backward(gradient, learning_rate)

        error /= len(image)
        print(f"{e + 1}/{epochs}, error={error}")
 
network = NeuralNetwork()

network.layers = [
    denselayer.DenseLayer(28 * 28, 10),
    activationandlosses.TanH(),
    denselayer.DenseLayer(28 * 28, 10),
    activationandlosses.TanH()
]

network.train(network.training_data, network.training_labels)