#testing the neural network on the Galaxy10 DECaLS dataset
#https://github.com/henrysky/Galaxy10

import h5py
import numpy as np
from p1 import compute_accuracy, Layer_Dense, Layer_Dropout,Activation_ReLU, Activation_LeakyReLU, Activation_Softmax_Loss_CategoricalCrossentropy, Optimizer_SGD, clip_gradients, Activation_Softmax, Loss_CategoricalCrossentropy, Optimizer_Adam



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

#normalize and flatten6
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
dense1 = Layer_Dense(numInputs, 128) #128 is the length of the output vector from this layer, which is also the number of neurons in this layer
activation1 = Activation_LeakyReLU()
dropout1 = Layer_Dropout(0.5)

dense2 = Layer_Dense(128, 64) #128 is the length of the output vector from this layer, which is also the number of neurons in this layer
activation2 = Activation_LeakyReLU()
dropout2 = Layer_Dropout(0.5)

dense3 = Layer_Dense(64, numClasses) #output layer here is 10 (numClasses)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.0001)

epochs = 20
batch_size = 64

if __name__ == "__main__":
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
            dropout1.forward(activation1.output, training=True)
            dense2.forward(activation1.output)
            activation2.forward(dense2.output)
            dropout2.forward(activation2.output, training=True)
            dense3.forward(activation2.output)

            loss = loss_activation.forward(dense3.output, y_batch)
            epochLoss += loss

            predictions = np.argmax(loss_activation.output, axis=1)
            epochAccuracy += np.mean(predictions == y_batch)
            
            #backward pass
            loss_activation.backward(loss_activation.output, y_batch)
            dense3.backward(loss_activation.dinputs)
            dropout2.backward(dense3.dinputs)
            activation2.backward(dense3.dinputs)
            dense2.backward(activation2.dinputs)
            dropout1.backward(dense2.dinputs)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)
            
            #clip gradients
            clip_gradients(dense1)
            clip_gradients(dense2)
            clip_gradients(dense3)

            #update weights and biases
            optimizer.update_params(dense1)
            optimizer.update_params(dense2)
            optimizer.update_params(dense3)
        
        dead_ratio = np.mean(activation1.output == 0)
        print(f"Fraction of dead ReLU outputs in layer 1: {dead_ratio:.2%}")
        print(f"Epoch {epoch+1:>2}/{epochs}  loss: {epochLoss/numBatches:.4f}  acc: {epochAccuracy/numBatches*100:.1f}%")
        print()

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
