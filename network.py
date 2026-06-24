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
    def calculate(self, output, y, class_weights=None):
        #y is the target values
        sample_losses = self.forward(output, y, class_weights = class_weights)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    #inherits from the Loss class
    def forward(self, y_pred, y_true, class_weights=None):
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
            sample_classes = y_true
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
            sample_classes=np.argmax(y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)

        if class_weights is not None:
            #scale each sample's loss by the weight of its true class
            sample_weights = class_weights[sample_classes]
            negative_log_likelihoods = negative_log_likelihoods * sample_weights

        return negative_log_likelihoods

class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true, class_weights=None):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true, class_weights=class_weights)
    
    def backward(self, dvalues, y_true, class_weights=None):
        samples = len(dvalues)
        if len(y_true.shape) == 2: #if one hot encoded, turn into discrete values
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy() #this is the gradient of the loss with respect to the inputs of the softmax function, which is also the output of the last dense layer
       
        #subtract 1 from the values of the correct labels
        #then divide by the number of samples to get the average gradient across the batch
        self.dinputs[range(samples), y_true] -= 1 

        if class_weights is not None:
            #scale each sample's gradient row by the weight of its true class.
            #only the forward loss value (above) has zero effect on the weight updates,
            #since backprop uses dinputs, not the scalar loss, to compute gradients.
            sample_weights = class_weights[y_true].reshape(-1, 1)
            self.dinputs = self.dinputs * sample_weights

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

def compute_class_weights(y_true, num_classes, max_weight=None):
    #some of the classes in NSL-KDD, specifically r2l and u2r, dont have many examples in the training set
    #this caused the network to never predict anything as r2l or u2r, since it didn't have enough examples to learn how they show up
    #this function will give the rarer classes a higher weight, and gradient, so that they are valued more
    #weight_c = total samples / (num_classes * count_c)
    counts = np.bincount(y_true, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1) #remove 0s so we dont divide by 0
    total = len(y_true)
    weights = total / (num_classes * counts)

    if max_weight != None:
        #capping the weight so it doesnt get tooo high either
        weights = np.minimum(weights, max_weight)
    return weights.astype(np.float32)

def compute_f1_per_class(predictions, y_true, num_classes):
    #returns precision, recall, f1 arrays (one value per class), useful for spotting
    #classes that look fine on overall accuracy but are actually being ignored
    
    #creating one slot per class
    predicted_labels = np.argmax(predictions, axis=1) #the index with the max value ex. [0.1,0.7,0.2] would return index 1
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    #iterating through each class
    for c in range(num_classes):
        tp = np.sum((predicted_labels == c) & (y_true == c)) #predicted c, was c
        fp = np.sum((predicted_labels == c) & (y_true != c)) #predicted c, was not c
        fn = np.sum((predicted_labels != c) & (y_true == c)) #not predicted c, was c

        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0 #when predicted c how many of those are right
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0 #out of all the true c's how many did the model predict
        
        #f1 is the mean of precision and recall
        f1[c] = (2 * precision[c] * recall[c] / (precision[c] + recall[c])
                 if (precision[c] + recall[c]) > 0 else 0.0)
    return precision, recall, f1

def clip_gradients(layer, max_norm=5.0):
    #will scale down gradients if they are too large
    norm = np.sqrt(np.sum(layer.dweights**2) + np.sum(layer.dbiases**2))
    if norm > max_norm:
        scale = max_norm / norm
        layer.dweights *= scale
        layer.dbiases *= scale


#save and load the model

def save_model(layers, path, model_name="model", extra_meta=None):
    arrays = {}
 
    # store each layer's parameters
    for i, layer in enumerate(layers):
        arrays[f"layer_{i}_weights"] = layer.weights
        arrays[f"layer_{i}_biases"]  = layer.biases
 
    # store metadata so load_model knows how to reconstruct
    arrays["num_layers"]   = np.array(len(layers), dtype=np.int32)
    arrays["model_name"]   = np.array(model_name)            # 0-d string array
    arrays["layer_shapes"] = np.array(
        [(l.weights.shape[0], l.weights.shape[1]) for l in layers], dtype=np.int32
    )   # shape: (num_layers, 2)  — each row is [n_inputs, n_neurons]
 
    # optional metadata (accuracy, loss, epoch count, etc.)
    if extra_meta:
        for key, value in extra_meta.items():
            arrays[f"meta_{key}"] = np.array(value)
 
    np.savez(path, **arrays)
    print(f"[save_model] saved {len(layers)} layer(s) → {path}.npz")
    if extra_meta:
        for k, v in extra_meta.items():
            print(f"             {k}: {v}")

def load_model(path):

    #need allow_pickle to be true bc model_name is a 0-d object array
    data = np.load(path, allow_pickle=True)
 
    num_layers   = int(data["num_layers"])
    model_name   = str(data["model_name"])
    layer_shapes = data["layer_shapes"]   # shape (num_layers, 2)
 
    layers = []
    for i in range(num_layers):
        n_inputs, n_neurons = int(layer_shapes[i, 0]), int(layer_shapes[i, 1])
 
        # build a fresh layer
        layer = Layer_Dense(n_inputs, n_neurons)
        layer.weights = data[f"layer_{i}_weights"].astype(np.float32)
        layer.biases  = data[f"layer_{i}_biases"].astype(np.float32)
        layers.append(layer)
 
    # collect metadata to hand back to the caller
    meta = {
        "model_name"  : model_name,
        "num_layers"  : num_layers,
        "layer_shapes": [(int(layer_shapes[i, 0]), int(layer_shapes[i, 1]))
                         for i in range(num_layers)],
    }
    for key in data.files:
        if key.startswith("meta_"):
            meta[key] = data[key].item()   # .item() converts 0-d array → Python scalar
 
    print(f"[load_model] loaded '{model_name}' — {num_layers} layer(s) from {path}")
    for i, (ni, nn) in enumerate(meta["layer_shapes"]):
        print(f"             layer {i}: ({ni} → {nn})")
 
    return layers, meta