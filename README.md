**End-to-End Neural Network Implementation in Python**<br>
**readme currently unfinished**

A completely connected neural network built using only NumPy and trained on the Fashion MNIST dataset. Every forward/backward pass, gradient, and weight update is done by hand without external libraries or functions, including the backpropagation alogrithim itself. 

Starting from random weights, the model reaches 85.6% training accuracy and 84.0% test accuracy.

<ins>Table of Contents</ins>
Dataset <br>
Network Architecture <br>
Forward Passes <br>
Loss Function <br>
Backpropagation <br>
Optimizer <br>
Training Loop <br>
Results <br>

<ins>DataSet</ins> 
The network is currently trained on the Fashion MNIST dataset, which is a replacement for the classic MNIST (handwritten digits) dataset. It is a collection of 70,000 total images of clothing articles, each belonging to one of 10 classes. The images have the same structure as MNIST: grayscale and 28x28 pixels. These are the 10 classes: 


| Label | Class |
| ------------- | ------------- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

Before training, the images were flattened. This means instead of a 2D array of 28x28, they were reshaped into a 1D array of 784 values. This makes it easier for the network's layers to consume data, and doesn't have any negative effect either. The image arrays were also normalized. This is the process of converting values to be between a range of [0.0, 1.0]. Orignally, since this dataset consists of images, the values for each example were between [0, 255]. Normalization keeps the inputes small and prevents erractic weight updates or unnesscary complexity later on during training. 

<ins>Network Architecture</ins> <br>
This network has three layers, with ReLU activation on the hidden layers and Softmax activation on the output. 

Input(784 values) <br>
&emsp;🡣 <br>
Hidden Layer 1 -- 128 neurons, ReLU activation <br>
&emsp;🡣 <br>
Hidden Layer 2 -- 64 neurons, ReLU activation <br>
&emsp;🡣 <br>
 Output Layer 3 -- 10 neurons, Softmax activation <br>
&emsp;🡣 <br>
    Output -- a probability distribution over 10 classes for each example <br>

The weights are initialized randomly through He initialization, which is where weights are taken from a normal distrubution that is scaled by $\sqrt{\frac{2}{n_inputs}}$. This type of intialization is specifically good for ReLU activations because it prevents vanishing or exploding gradients during training. In other words, this helps keep values flowing through the network without shrinking to 0 or excessivly growing after going through lots of forward/backward passes. The biases are all initialized to 0. 

<ins>Forward Pass</ins> <br>
Each neuron does this computation: output_of_neuron = input\*weight + bias. Another way to represent this is through matrix multiplication. If we combine all the inputs, weights, and biases for a specific layer into matrices, we can do output_of_layer = inputs_matrix\*weights_matrix + biases matrix. Note that the inputs_matrix and weights_matrix are matrix multiplied(not the same as regular multiplication).

The *ReLU* activation function is applied after each hidden layer: $ReLU(x) = max(0,x)$ <br>
The *Softmax* activation function is applied to the output layer: $Softmax(x) = \frac{e^x}{\text{sum of e**y for each output of that layer}}$
Softmax activation prevents numerical overflow by subtracting the minimum value of a set of inputs(these inputs to the softmax function are the outputs of a whole layer) from all values in the set.

<ins>Loss</ins> <br>
The network trains using *Categorical Cross-Entropy* loss: loss = -(one_hot[i]*log(prediction[i])) for i in range (0, num_classes)<br>

Lets break this down: 
* each training example has its own loss
* one_hot[i] refers to the one-hot encoding for a certain class, where the other classes are 0 and the correct class is 1. For example, the one-hot encoding for class 2 would be: [0,0,1,0,0,0,0,0,0,0]
* prediction[i] is the vector of how likely an example is of a certain class.
* thanks to the one-hot encoding multiplying all wrong classes by 0, what we are essentially doing is: $-log(prediction_probability_for_correct_class)$

<ins>Backpropagation</ins> <br>
Backpropagation is the process of going backwards through the network(starting from output layer to the inputs) and computing how much each weight and bias contribute to the loss (here we use Categorical Cross-Entropy Loss. By doing this, we get an idea of how to tweak that weight/bias in order to minimize loss. <br>
At the core of this process is the chain rule. Remember that for a single layer, $z = Wx + b$, where $z$ is output, $W$ is weight, $x$ is input, and $b$ is bias. We then apply the activation function and calculate loss. What is we want to know is $\frac{dL}{dW}$ - how much will loss change if we change a specific weight? Thanks to the chain rule that expression can be broken down into: $\frac{dL}{dW} = \frac{dL}{da} \* \frac{da}{dz} \* \frac{dz}{dW}$.  $\frac{dL}{da}$ is how the activation affects the loss, $\frac{da}{dz}$ is how the activation changes in response to the input, and $\frac{dz}{dW}$ is how the weights affect the output. With $\frac{dL}{dW}$, we can then do $new_weight = old_weight + (learning_rate)\frac{dL}{da}$. Each class has a backward() method that implements this backpropagation process.  <br>
For the Softmax activation and Cross-Entropy Loss, the gradient simplifies to $gradient = output_prediction - output_true$. This is essentially the same as $predicted probability - 1$. This tells the network to push the correct class's probability to 1 and others to 0.<br>
The ReLU function backwards ends up as a step function. Recall that $ReLU(x) = max(x,0)$, so $\frac{d_ReLU}{dx} = 1$ if $x > 0$, or 0 otherwise. When going backwards, this means any neuron that was inactive during forward pass (thanks to ReLU) is zeroed out. <br>
For each layer, 3 calculations are done to update the weights, biases, and send to the previous layer how much the inputs affected the loss. The gradient for the weights was $d_weights = inputs.T \* d_values$. We take the transpose of inputs and matrix multiply by d_values, the measure of how much the ouputs to the current layer affect the loss. The gradient for the biases is $d_biases = sum(d_values, axis=0)$, or in other words just the sum of how much each neuron output contributed to the loss. The gradient of the inputs is $d_inputs = d_values \* weights.T$, or d_values matrix multiplyed by the transpose of the weights.





References: <br>
[https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) <br>
[https://towardsdatascience.com/kaiming-he-initialization-in-neural-networks-math-proof-73b9a0d845c4/](https://towardsdatascience.com/kaiming-he-initialization-in-neural-networks-math-proof-73b9a0d845c4/) <br>
[https://www.geeksforgeeks.org/deep-learning/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/](https://www.geeksforgeeks.org/deep-learning/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/) <br>
[https://www.geeksforgeeks.org/deep-learning/categorical-cross-entropy-in-multi-class-classification/](https://www.geeksforgeeks.org/deep-learning/categorical-cross-entropy-in-multi-class-classification/) <br>
[https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)<br>
