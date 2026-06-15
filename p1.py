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
        self.inputs = inputs #save the inputs for backpropagation
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues, l2_lambda=0.001):
        #dvalues is the gradient of the loss with respect to the output of this layer, which is also the input to the next layer
        
        #gradient of the loss with respect to the weights
        #2 * l2_lambda * self.weights adds the L2 penalty term, pushing large weights downward during training
        #by discoraging very large weights we help prevent overfitting
        self.dweights = np.dot(self.inputs.T, dvalues) + 2 * l2_lambda * self.weights
        #gradient of the loss with respect to the biases
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) 
        #gradient of the loss with respect to the inputs of this layer, which is also the output of the previous layer
        self.dinputs = np.dot(dvalues, self.weights.T) 

class Activation_ReLU:
    #ReLU = rectifity linear unit
    def forward(self, inputs):
        self.inputs = inputs #save the inputs for backpropagation
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        #if input value is less than or equal to 0, set the gradient to 0, otherwise keep it the same
        self.dinputs = dvalues.copy() #copy the values of dvalues so we can modify them without affecting the original array
        self.dinputs[self.inputs <= 0] = 0
class Activation_LeakyReLU: 
    def __init__(self, alpha=0.01):
        self.alpha = 0.01
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] *= self.alpha

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

class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2: #if one hot encoded, turn into discrete values
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy() #this is the gradient of the loss with respect to the inputs of the softmax function, which is also the output of the last dense layer
       
        #subtract 1 from the values of the correct labels
        #then divide by the number of samples to get the average gradient across the batch
        self.dinputs[range(samples), y_true] -= 1 
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

class Optimizer_Adam:
    #adam optimization keeps a running avg of past gradients, and uses that to update current gradient
    #it also keeps a running avg of squared gradients per weight
    #then it divides the update by sqrt(v) where v is the running avg of squared gradients
    #ex.
        #w = 1
        #learning_rate = 0.1
        #gradient = 0.5
        #m = 0, v = 0 (m and v are the running avgs of past gradients and squared gradients)
        
        #after 1st update:
        #m = 0.9*m + 0.1*gradient
        #v = 0.999*v + 0.001*gradient^2
        #newWeight = w - (learning_rate*m)/(sqrt(v)+1e-7) #we add 1e-7 to avoid division by 0
    
    #this way, weights with large gradients will have their updates scaled down, and weights with small gradients will have their updates scaled up, which can help us avoid vanishing/exploding gradients and can lead to faster convergence
    #in other words, it helps us take bigger steps in the right direction, and smaller steps in the wrong direction

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1 #past gradient * beta_1 
        self.beta_2 = beta_2 #past squared gradient * beta_2
        self.epsilon = epsilon
        self.t = 0 #timestep counter
    
    def update_params(self, layer):
        #this will be called each layer during each update step
        if not hasattr(layer, 'm_weights'):
            layer.m_weights = np.zeros_like(layer.weights)
            layer.v_weights = np.zeros_like(layer.weights)
            layer.m_biases = np.zeros_like(layer.biases)
            layer.v_biases = np.zeros_like(layer.biases)
        #update timestep
        self.t += 1

        #update running avgs
        layer.m_weights = self.beta_1 * layer.m_weights + (1 - self.beta_1) * layer.dweights #dweights is the current gradient of the loss with respect to the weights, which is calculated during backpropagation
        layer.v_weights = self.beta_2 * layer.v_weights + (1 - self.beta_2) * layer.dweights**2
        layer.m_biases  = self.beta_1 * layer.m_biases  + (1 - self.beta_1) * layer.dbiases
        layer.v_biases  = self.beta_2 * layer.v_biases  + (1 - self.beta_2) * layer.dbiases**2

        #correct bias
        #this is because the running avgs are initialized to 0, so they are biased towards 0 at the beginning of training, especially when beta_1 and beta_2 are close to 1
        m_w = layer.m_weights / (1 - self.beta_1**self.t)
        v_w = layer.v_weights / (1 - self.beta_2**self.t)
        m_b = layer.m_biases  / (1 - self.beta_1**self.t)
        v_b = layer.v_biases  / (1 - self.beta_2**self.t)

        #update weights and biases
        layer.weights -= self.learning_rate * m_w / (np.sqrt(v_w) + self.epsilon) #doing - to move in the direction of the negative gradient, which is the direction of steepest descent
        layer.biases  -= self.learning_rate * m_b / (np.sqrt(v_b) + self.epsilon)
class Layer_Dropout:
    #during training, this will randomly turn off some of the inputs
    #in other words, during training(forward and backward) some neurons are essentially ignored
    #this helps prevent the model from just memorizing the training set instead of actually learning
    def __init__(self, rate):
        self.rate=rate
    def forward(self, inputs, training=True):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        #random mask of 0s and 1s
        #each neuron is kept with the probability of 1-rate
        
        self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)
        self.output = inputs * self.mask
    def backward(self, dvalues):
        self.dinputs = dvalues * self.mask


def compute_accuracy(predictions, y_true):
    #fraction of samples where argmax prediction matches the true label
    predicted_labels = np.argmax(predictions, axis=1)
    return np.mean(predicted_labels == y_true)
def clip_gradients(layer, max_norm=5.0):
    #will scale down gradients if they are too large
    norm = np.sqrt(np.sum(layer.dweights**2) + np.sum(layer.dbiases**2))
    if norm > max_norm:
        scale = max_norm / norm
        layer.dweights *= scale
        layer.dbiases *= scale

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

#the gradient of the loss with respect to weight 
#positive gradient means that increasing the weight will increase the loss, so we want to decrease the weight
#^^vice versa for negative gradient

#update rule is weight = weight - learning_rate * gradient


loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(learning_rate=0.01)

epochs = 20
batch_size = 256

if __name__ == "__main__":
    for epoch in range(epochs):
        #each epoch will use a different set of data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = len(X_train) //batch_size

        for i in range(num_batches):
            #now we get a different batch of data for each iteration of the loop
            X_batch = X_shuffled[i*batch_size:(i+1)*batch_size] 
            y_batch = y_shuffled[i*batch_size:(i+1)*batch_size]

            #forward
            dense1.forward(X_batch)
            activation1.forward(dense1.output)
            dense2.forward(activation1.output)
            activation2.forward(dense2.output)
            dense3.forward(activation2.output)
            loss = loss_activation.forward(dense3.output, y_batch)
            epoch_loss += loss

            #accuracy
            predictions = np.argmax(loss_activation.output, axis=1)
            accuracy = np.mean(predictions == y_batch)
            epoch_accuracy += accuracy

            #backward
            loss_activation.backward(loss_activation.output, y_batch)
            dense3.backward(loss_activation.dinputs)
            activation2.backward(dense3.dinputs)
            dense2.backward(activation2.dinputs)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)

            #updating the optimizer
            optimizer.update_params(dense1)
            optimizer.update_params(dense2)
            optimizer.update_params(dense3)

        print(f"Epoch {epoch+1:>2}/{epochs}  loss: {epoch_loss/num_batches:.4f}  acc: {epoch_accuracy/num_batches*100:.1f}%")
        if (epoch == 19):
            print(f"Final Accuracy(on training data): {epoch_accuracy/num_batches*100:.1f}") 


    #testing on X_text y_test
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    loss_activation.activation.forward(dense3.output)  # softmax only, no loss needed
    
    loss_fn   = Loss_CategoricalCrossentropy()
    test_loss = loss_fn.calculate(loss_activation.activation.output, y_test)
    test_acc  = compute_accuracy(loss_activation.activation.output, y_test)
    
    print()
    print(f"\n── Test set evaluation ──")
    print(f"Loss:     {test_loss:.4f}")
    print(f"Accuracy: {test_acc*100:.1f}%")
