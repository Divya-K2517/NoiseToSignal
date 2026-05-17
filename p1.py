import sys
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib
from keras.datasets import fashion_mnist

#this file is representing one layer of 3 neurons

np.random.seed(0)

#2 hidden layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2/n_inputs) 
            #we scale by (2/n_inputs) because this is the recommended way to initialize weights for ReLU activation function, which we are using in our hidden layers, and it helps us avoid vanishing/exploding gradients
            #vanishing/exploding gradients is when the gradients() become too small or too large during backpropagation
            #gradients are the derivatives of the loss function with respect to the weights and biases, and they are used to update the weights and biases during training
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

def accuracy(predictions, y_true):
    #fraction of samples where argmax prediction matches the true label
    predicted_labels = np.argmax(predictions, axis=1)
    return np.mean(predicted_labels == y_true)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
#dataset: https://github.com/zalandoresearch/fashion-mnist
#has 10 classes of clothing items, 60,000 training examples, 10,000 test examples
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#X_train.shape is (60000,28,28) => 60000 pic, each pic is a 28x28 pixel image, this is like a 3D array, we have 60000 2D arrays of 28x28
#y_train.shape is (60000,) => 60000 labels, each label is a number from 0-9 representing the class of clothing
#we need to flatten the 28x28 images into a 1D array of 784 pixels
X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

#we also need to normalize the pixel values to be between 0 and 1, instead of 0-255
X_train /= 255.0
X_test /= 255.0

dense1 = Layer_Dense(784,128) #784 inputs bc the data is784 pixels and we choose 128 neurons in hidden1 layer
activation1 = Activation_ReLU()

dense2 = Layer_Dense(128, 64) #128 inputs bc the outputs of the last layer was 128 neurons, and we choose 64 neurons in hidden2 layer
activation2 = Activation_ReLU()

dense3 = Layer_Dense(64, 10) #64 inputs bc the outputs of the last layer was 64 neurons, and we choose 10 neurons in output layer bc we have 10 classes
activation3 = Activation_Softmax()

#FORWARD PASS 
#on just a batch of data, not the whole dataset
BATCH_SIZE = 256
X_batch = X_train[:BATCH_SIZE]
y_batch = y_train[:BATCH_SIZE]

dense1.forward(X_batch)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

dense3.forward(activation2.output)
activation3.forward(dense3.output)

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
loss = loss_function.calculate(activation3.output, y_batch) #activation3.output is softmax outputs, y_batch is the target
accuracy = accuracy(activation3.output, y_batch)


#loss tells us how wrong smth is
#how to decrease loss - optimize weights and biases
#make random tweaks to weights and biases (ex: add a random number to them)
#and then see if loss decreases
#^^^ but even this is not optimal for more complex datasets

#BACKPROPAGATION
#for each weight and bias, we can also change only one at a time and calculate the resulting loss

#the magnitude of the change in loss will be the gradient of the loss with respect to that weight or bias
#this tells us how sensitive the loss is to changes in that weight or bias

#ex:
    #if we are looking at an example of 2, our inital predictions will be pretty spread out across all 10 classes
    #the goal would be to increase the activation of the 2nd neuron in the output layer, and decrease the activation of all other neurons in the output layer
    #remember the activation is reLU(dot product of inputs(outputs from previous layer) and weights (for that layer) + bias)
    #we should increase weights in proportion to activations, so if a neuron has a higher activation increase weights more?

#^^^^continuing from the above example
#the desire of the 2 example for how the activations should change may be one number
#we add up the desires of all the other neurons in that output layer(which all want to be less active bc they ae not 2)
#now after back propagating through all the layers, we know what the 2 wants for each weight
#then average the desired changes for each weight across all 10 classes to get the desired change for each weight

#^^that process takes a long time for computers, so:
#divide training data into mini-batches and do backprop for each batch
#what next?

#DISPLAYING OUTPUTS
print(f"\n── Forward pass on {BATCH_SIZE} samples ──")
print(f"Loss:     {loss:.4f}    This means {np.exp(-loss):.4f} is the probability of being correct  (random init baseline ≈ {np.log(10):.4f})")
print(f"Accuracy: {accuracy*100:.1f}%  (random init baseline ≈ 10%)")
print(f"\nFirst 5 predictions:")
for i in range(5):
    pred  = np.argmax(activation3.output[i])
    true  = y_batch[i]
    print(f"  Sample {i}: predicted={CLASS_NAMES[pred]:<15} true={CLASS_NAMES[true]}")