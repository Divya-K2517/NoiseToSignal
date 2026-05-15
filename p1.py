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

class Loss:
    def calculate(self, output, y):
        #y is the target values
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    #inherits from the Loss class
    def forward(self, y_pred, y_true):
        samples = len(y_pred) #length of the batch
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) #clipping the data to avoid log(0) which is undefined

        if (len(y_true.shape)) == 1:
            #this means they passed scalar values, not one-hot encoded values
            correct_confidences = y_pred_clipped[range(samples), y_true]
                #^^^^^^ what this line does:
                #softmax_output = [[0.7,0.1,0.2],
                #                  [0.1,0.5,0.4],
                #                  [0.02,0.9,0.08]]
                #range(samples) would be [0,1,2]
                #y_true = [0,1,1] => vertical
                #
                #confidences on correct labels = [0.7,0.5,0.9] 
        elif len(y_true.shape) == 2:
            #in this case its a one hot encoded vector
            correct_confidences = np.sum(y_pred_clipped *y_true, axis=1)
                #^^^^^^ what this line does:
                #softmax_output = [[0.7,0.1,0.2],
                #                  [0.1,0.5,0.4],
                #                  [0.02,0.9,0.08]]
                #range(samples) would be [0,1,2]
                #y_true = [[1,0,0],
                #          [0,1,0],
                #          [0,1,0]]
                #
                #confidences on correct labels = softmax*y_true = 
                    #   = [[0.7, 0, 0],
                    #      [0.0, 0.5, 0.0],
                    #      [0.0, 0.9, 0.0]]
                #and np.sum(^) = [0.7,
                #                 0.5,
                #                 0.9]
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods



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

#one hot encoding
#vector that is n long, where n is number of classes
#the vector is filled with 0's except for the index of your target class, which will be 1
#prediction is the vector of predictions, or the output from your output layer of the neural network
#ex:
    #class = 5
    #label = 2
    #one-hot: [0,0,1,0,0]
    #prediction: [0.3,0.1,0.4,0.1, 0.1]

#loss function is -sum(one-hot[i]*log(prdiction[i])) for i in range (0,num_classes)

# we would have a batch of softmax outputs and an array of targets
#ex:
    #softmax_output = [[0.7,0.1,0.2],
    #                  [0.1,0.5,0.4],
    #                  [0.02,0.9,0.08]]
    #class_targets = [0,1,1] => vertical
    #
    #confidences on correct labels = [0.7,0.5,0.9] 
        #0.7 bc its the 0th element, and 0.5 and 0.9 are the 1st element

    
loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y) #activation2.output is softmax outputs, y is the target
print("Loss: ", loss)


#loss tells us how wrong smth is
#how to decrease loss - optimize weights and biases
#make random tweaks to weights and biases (ex: add a random number to them)
#and then see if loss decreases
#^^^ but even this is not optimal for more complex datasets

#for each weight and bias, we can also change only one at a time and calculate the resulting loss

