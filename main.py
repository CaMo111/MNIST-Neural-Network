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
        
        #training data has 59,000 entries for each list, each representing
        #the ith pixel, from 0 to 783 index
        # now need to normalise pixels by Scaling Pixel Values by 1/255
        #each column represents a full unique image <-- 
        #eg matrix of data looks like 
        '''
        [
        [0th image, 1st image.... 58,999 image] 0th pixel
        [0th image, 1st image.... 58,999 image] 1st pixel
        ...
        [0th image, 1st image.... 58,999 image] 783th pixel
        ]
        '''
        print(len(dev_data))
        for ith_pixel in dev_data:
            for unique in ith_pixel:
                unique = (1/255) * unique

        for ith_pixel in training_data:
            for unique in ith_pixel:
                unique = (1/255) * unique

        return dev_labels, dev_data, training_labels, training_data
    
    def predict(self, input):
        output = input
        for layer in network.layers:
            output = layer.forward(output)
        return output

    def train(self, loss, lossprime, image_or_data, label, epochs=100, learning_rate=0.1):
        for e in range(epochs):
            error = 0
 
network = NeuralNetwork()

network.layers = [
    denselayer.DenseLayer(28 * 28, 10),
    activationandlosses.TanH(),
    denselayer.DenseLayer(28 * 28, 10),
    activationandlosses.TanH()
]