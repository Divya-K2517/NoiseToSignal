#testing the neural network on the Galaxy10 DECaLS dataset
#https://github.com/henrysky/Galaxy10

import h5py
import numpy as np
from p1 import compute_accuracy, Layer_Dense, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy, Optimizer_SGD, Activation_Softmax, Loss_CategoricalCrossentropy



CLASS_NAMES = [
    "Disk, Face-on, No Spiral",
    "Smooth, Completely round",
    "Smooth, in-between round",
    "Smooth, Cigar shaped",
    "Disk, Edge-on, Rounded Bulge",
    "Disk, Edge-on, Boxy Bulge",
    "Disk, Edge-on, No Bulge",
    "Disk, Face-on, Tight Spiral",
    "Disk, Face-on, Medium Spiral",
    "Disk, Face-on, Loose Spiral"
]

# To get the images and labels from file
with h5py.File('Galaxy10_DECals.h5', 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])

#normalize and flatten
images = images.reshape(images.shape[0], -1).astype(np.float32) #this makes them 1D vectors
images /= 255.0 #normalize to a range of [0,1]

#split into test and train
np.random.seed(0)
indices = np.random.permutation(len(images)) #getting random indices to shuffle the data
split = int(0.8 * len(images)) #80% for training, 20% for testing
trainIdx, testIdx = indices[:split], indices[split:]

x_train, y_train = images[trainIdx], labels[trainIdx]
x_test, y_test = images[testIdx], labels[testIdx]

numInputs = x_train.shape[1] #number of features (pixels)
numClasses = len(CLASS_NAMES) #number of classes

#define the layers of the neural network
dense1 = Layer_Dense(numInputs, 256) #256 is the length of the output vector from this layer, which is also the number of neurons in this layer
activation1 = Activation_ReLU()

dense2 = Layer_Dense(256, 128) #128 is the length of the output vector from this layer, which is also the number of neurons in this layer
activation2 = Activation_ReLU()

dense3 = Layer_Dense(128, numClasses) #output layer here is 10 (numClasses)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(learning_rate=0.01)

epochs = 20
batch_size = 64

for epoch in range(epochs):
    #each epoch we train a single batch

    #shulffe the training data
    indices = np.random.permutation(len(x_train))
    x_shuffled, y_shuffled = x_train[indices], y_train[indices]

    epochLoss = 0
    epochAccuracy = 0
    numBatches = len(x_train) // batch_size

    for i in range(numBatches):
        #to get the batch, we do the 1, 2, 3, ... batch_size, then the next batch_size, etc
        x_batch = x_shuffled[i*batch_size:(i+1)*batch_size]
        y_batch = y_shuffled[i*batch_size:(i+1)*batch_size]

        #forward pass
        dense1.forward(x_batch)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        dense3.forward(activation2.output)
        loss = loss_activation.forward(dense3.output, y_batch)
        epochLoss += loss

        predictions = np.argmax(loss_activation.output, axis=1)
        epochAccuracy += np.mean(predictions == y_batch)
        
        #backward pass
        loss_activation.backward(loss_activation.output, y_batch)
        dense3.backward(loss_activation.dinputs)
        activation2.backward(dense3.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        #update weights and biases
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)
    
    print(f"Epoch {epoch+1:>2}/{epochs}  loss: {epochLoss/numBatches:.4f}  acc: {epochAccuracy/numBatches*100:.1f}%")


#testing the model on the test set
dense1.forward(x_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)

activation3 = Activation_Softmax()
activation3.forward(dense3.output)

loss_fn = Loss_CategoricalCrossentropy()
test_loss = loss_fn.calculate(activation3.output, y_test)
test_acc = compute_accuracy(activation3.output, y_test)

print(f"\n── Test set evaluation ──")
print(f"Loss:     {test_loss:.4f}")
print(f"Accuracy: {test_acc*100:.1f}%")
