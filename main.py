import numpy as np 
import pandas as pd

#each row is image, 0th index is the label but 1 to 784 is the pixels
def process_data(filepath_or_buffer: str):
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
    
    #each column represents a full unique image
    for ith_pixel in dev_data:
        for unique in ith_pixel:
            unique = (1/255) * unique

    for ith_pixel in training_data:
        for unique in ith_pixel:
            unique = (1/255) * unique

    return dev_labels, dev_data, training_labels, training_data
 
dev_labels, dev_data, training_labels, training_data = process_data("mnist_train.csv")

print(dev_data)

def initialise_parameters(training_labels, training_data):
    W1 = 1 
    b1 = 1 
    W2 = 1 
    b2 = 1
    

