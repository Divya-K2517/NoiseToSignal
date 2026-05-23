**End-to-End Neural Network Implementation in Python**

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
      🡣 <br>
Hidden Layer 1 -- 128 neurons, ReLU activation <br>
      🡣 <br>
Hidden Layer 2 -- 64 neurons, ReLU activation <br>
      🡣 <br>
 Output Layer 3 -- 10 neurons, Softmax activation <br>
      🡣 <br>
    Output -- a probability distribution over 10 classes for each example <br>

The weights are initialized randomly through He initialization, which is where weights are taken from a normal distrubution that is scaled by $\sqrt{$\frac{2}{n_inputs}$}$. This type of intialization is specifically good for ReLU activations because it prevents vanishing or exploding gradients during training. In other words, this helps keep values flowing through the network without shrinking to 0 or excessivly growing after going through lots of forward/backward passes. The biases are all initialized to 0. 

<ins>Forward Pass</ins> <br>
Each neuron does this computation: output_of_neuron = input\*weight + bias. Another way to represent this is through matrix multiplication. If we combine all the inputs, weights, and biases for a specific layer into matrices, we can do output_of_layer = inputs_matrix\*weights_matrix + biases matrix. Note that the inputs_matrix and weights_matrix are matrix multiplied(not the same as regular multiplication).

The *ReLU* activation function is applied after each hidden layer: ReLU(x) = max(0,x)
The *Softmax* activation function is applied to the output layer: Softmax(x) = $\frac{$e^x$}{sum_of_$e^y$_for_each_output_of_that_layer}$
Softmax activation prevents numerical overflow by subtracting the minimum value of a set of inputs(these inputs to the softmax function are the outputs of a whole layer) from all values in the set.




References:
[https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
[https://towardsdatascience.com/kaiming-he-initialization-in-neural-networks-math-proof-73b9a0d845c4/](https://towardsdatascience.com/kaiming-he-initialization-in-neural-networks-math-proof-73b9a0d845c4/)
[https://www.geeksforgeeks.org/deep-learning/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/](https://www.geeksforgeeks.org/deep-learning/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/)
