import sys
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib

#this file is representing one layer of 3 neurons

#np.random.seed(0)
nnfs.init() #sets a default data type for numpy


#2 hidden layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
            #weights is a matrix of n_inputs rows, n_neurons columns
        self.biases = np.zeros((1,n_neurons))
            #rn biases are intialized to zeros
            # if we end up with 0's as the output for many neurons in many layers, may have to increase default biases above 0
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    #ReLU = rectifity linear unit
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
class Activation_Softmax:
    def forward(self, inputs):
        #inputs are in a batch
        #we do inputs-max(inputs) so that all of our values become between 0-1 after exponentiating
        #the subtraction makes the max value 0, and all others negative
        #this helps us avoid overflow error
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        probabiltities = exp_values / np.sum(exp_values, axis=1, keepdims=True) #normalizing all values
        self.output = probabiltities


X, y = spiral_data(samples=100, classes=3) #100 samples per class and 3 seperate classes
dense1 = Layer_Dense(2,3) #2 inputs bc the spiral data is (x,y) data so its 2 coordinates, we choose 3 outputs can choose anything
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3) #3 inputs bc the outputs of the last layer was 3, and 3 outputs we choose bc 3 classes
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

