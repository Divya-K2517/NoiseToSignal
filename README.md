# **💡End-to-End Neural Network Implementation in Python**<br>
![Python](https://img.shields.io/badge/Python-3.x-blue)
![NumPy](https://img.shields.io/badge/NumPy-green)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-84%25-brightgreen)

<p align="center">
  <img src="fashion_mnist_confusion_matrix.png" width="44%" />
  <img src="nslkdd_confusion_matrix.png" width="44%" />
</p>

A completely connected neural network built using only NumPy and trained on the Fashion MNIST dataset. Every forward/backward pass, gradient, and weight update is done by hand without external libraries or functions, including the backpropagation alogrithim itself. 

Starting from random weights, the model reaches 85.3% training accuracy and 83.8% test accuracy on the Fashion MNIST dataset, and 93.4% training accuracy and 78.2% test accuracy on the NSL-KDD dataset.

## Table of Contents
**Datasets** <br>
**Network Architecture**<br>
**Forward Pass**<br>
**Loss Function**<br>
**Backpropagation**<br>
**Optimizer**<br>
**Regularization** <br>
**Gradient Clipping** <br>
**Training Loop**<br>
**Class Imbalance** <br>
**Results**<br>

## 👗 Dataset
The network was tested on two datasets: Fashion MNIST(a replacement for the classic MNIST handwritten digits), and NSL-KDD(an improved version of the KDD'99 set which categorizes network intrusions).

### Fashion MNIST
Fashion MNIST dataset is a collection of 70,000 total images of clothing articles, each belonging to one of 10 classes. The images have the same structure as MNIST: grayscale and 28x28 pixels. These are the 10 classes: 


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

### NSL-KDD (Network Intrusion Detection)
NSL-KDD is a tabular dataset, containing ~125,000 training and ~22,000 test records/rows with 41 features per record. There are 38 numeric columns, and 3 categorical text columns. The 3 text columns were one-hot encoded, and the other 38 were scaled to a range of [0,1]. These records are grouped into 22 specific types of attacks, but these can be collapsed into 5 broad categories. These are the 5 classes: 

| Label | Class | Description |
| ------------- | ------------- | ------------- |
| 0 | normal | actual traffic |
| 1 | dos | denial of service |
| 2 | probe | surveillance/scanning |
| 3 | r2l | remote-to-local exploit |
| 4 | u2r | user-to-root exploit |

It's worth noting that this dataset is a bit imbalanced when it comes to how many samples there are of certain classes. Namely, r2l and u2r make up 1-2% of the training data each, which has a large affect on training.

## 🕸️ Network Architecture
This network has three layers, with activation functions on the hidden layers and Softmax activation on the output. The sizes of each layer vary by dataset.

| Layer | Fashion MNIST Size | NSL-KDD Size | Activation |
|-------|------|------|------------|
| Input | 784 | 122 (38 numeric, 84 one-hot) | — |
| Hidden 1 | 128 neurons | 128 neurons | ReLU/LeakyReLU |
| Hidden 2 | 64 neurons | 64 neurons | ReLU/LeakyReLU |
| Output | 10 neurons | 5 neurons | Softmax |

The weights are initialized randomly through He initialization, which is where weights are taken from a normal distrubution that is scaled by $\sqrt{\frac{2}{n_inputs}}$. This type of intialization is specifically good for ReLU activations because it prevents vanishing or exploding gradients during training. In other words, this helps keep values flowing through the network without shrinking to 0 or excessivly growing after going through lots of forward/backward passes. The biases are all initialized to 0. 

## ➡️ Forward Pass
Each neuron does this computation: $$outputOfNeuron = input\*weight + bias $$. Another way to represent this is through matrix multiplication. If we combine all the inputs, weights, and biases for a specific layer into matrices, we can do $$outputOfLayer = inputsMatrix\*weightsMatrix + biasesMatrix$$. Note that the inputsMatrix and weightsMatrix are matrix multiplied(not the same as regular multiplication).

The *ReLU* activation function is applied after each hidden layer for the Fashion MNIST dataset: $$ReLU(x) = max(0,x)$$<br> 
The *Leaky ReLU* activation function is applied afer each hidden layer for the NSL-KDD dataset: $$\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ 0.01x & x \le 0 \end{cases}$$ <br>
Leaky ReLU was used to avoid the dying ReLU problem, which is when neurons using ReLU activation die out and remain at 0 in repsonse to a negative input. Leaky ReLU avoids this by giving a small non-zero gradient to negative inputs.
The *Softmax* activation function is applied to the output layer: $$Softmax(x) = \frac{e^x}{\text{sum of e**y for each output of that layer}}$$ <br>
Softmax activation prevents numerical overflow by subtracting the minimum value of a set of inputs(these inputs to the softmax function are the outputs of a whole layer) from all values in the set.


## 📉 Loss
The network trains using *Categorical Cross-Entropy* loss:  $$\mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)$$ <br>

Lets break this down: 
* each training example has its own loss
* one_hot[i] refers to the one-hot encoding for a certain class, where the other classes are 0 and the correct class is 1. For example, the one-hot encoding for class 2 would be: [0,0,1,0,0,0,0,0,0,0]
* prediction[i] is the vector of how likely an example is of a certain class.
* thanks to the one-hot encoding multiplying all wrong classes by 0, what we are essentially doing is: $-log(predictionProbabilityForCorrectClass)$ <br>

For NSL-KDD, this loss is scaled per-sample by a class weight, so that mistakes on some of the rarer classes like r2l and u2r will contribute more to the loss and overall matter more.


 
 ## ⬅️Backpropagation
Backpropagation is the process of going backwards through the network(starting from output layer to the inputs) and computing how much each weight and bias contribute to the loss (here we use Categorical Cross-Entropy Loss. By doing this, we get an idea of how to tweak that weight/bias in order to minimize loss. <br>

At the core of this process is the chain rule. Remember that for a single layer, $z = Wx + b$, where $z$ is output, $W$ is weight, $x$ is input, and $b$ is bias. We then apply the activation function and calculate loss. What is we want to know is $\frac{dL}{dW}$ - how much will loss change if we change a specific weight? Thanks to the chain rule that expression can be broken down into: $\frac{dL}{dW} = \frac{dL}{da} \* \frac{da}{dz} \* \frac{dz}{dW}$.  $\frac{dL}{da}$ is how the activation affects the loss, $\frac{da}{dz}$ is how the activation changes in response to the input, and $\frac{dz}{dW}$ is how the weights affect the output. With $\frac{dL}{dW}$, we can then do $new_weight = oldWeight + (learningRate)\frac{dL}{da}$. Each class has a backward() method that implements this backpropagation process.  <br>

For the Softmax activation and Cross-Entropy Loss, the gradient simplifies to $gradient = output_prediction - output_true$. This is essentially the same as $predicted probability - 1$. This tells the network to push the correct class's probability to 1 and others to 0.<br>

The ReLU function backwards ends up as a step function. Recall that $ReLU(x) = max(x,0)$, so $\frac{d_ReLU}{dx} = 1$ if $x > 0$, or 0 otherwise. When going backwards, this means any neuron that was inactive during forward pass (thanks to ReLU) is zeroed out. For Leaky ReLU, the negative branch creates a small gradient of $0.01$ instead of $0$, preventing neurons from getting stuck at an output of 0.<br>

For each layer, 3 calculations are done to update the weights, biases, and send to the previous layer how much the inputs affected the loss. The gradient for the weights was $dWeights = inputs.T \* dValues$. We take the transpose of inputs and matrix multiply by dValues, the measure of how much the ouputs to the current layer affect the loss. The gradient for the biases is $dBiases = sum(dValues, axis=0)$, or in other words just the sum of how much each neuron output contributed to the loss. The gradient of the inputs is $dInputs = dValues \* weights.T$, or dValues matrix multiplyed by the transpose of the weights. 

## 🔨Optimizer

### SGD (Stochastic Gradient Descent)
SGD is used for Fashion MNIST with a learning rate of 0.01. After the backpropagation step computes the gradients (how much the loss changes in response to weights, biases, and inputs), the optimizer applies these rules to every layer: <br>
$$weights = weights - (learningRate \* dweights) $$ <br>
$$biases = biases - (learningRate \* dbiases) $$ <br>

These equations move the weights/biases in the opposite direction of the gradient. For example, if the gradient is positive(which means a highet weight/bias produces greater loss), that means that the weight/bias will decrease(or become less positive). This allows loss to decrease. <br>

The learning rate is important because if it is too large, the updates will be an overshoot. Too small and the training will be very slow. 

### Adam (Adaptive Moment Estimation)
Adam is used for the NSL-KDD dataset. Instead of a fixed learning rate, Adam tracks two running averages per weight: 
- $m$ — an exponential moving average of past gradients (momentum): $m = \beta_1 m + (1 - \beta_1) \cdot dW$ <br>
- $v$ — an exponential moving average of past squared gradients: $v = \beta_2 v + (1 - \beta_2) \cdot dW^2$ <br>
Additionally, they are both bias-corrected to compensate for being initialized at zero:
$$\hat{m} = \frac{m}{1 - \beta_1^t} \quad \quad \hat{v} = \frac{v}{1 - \beta_2^t}$$ <br>
The update then divides by $\sqrt{\hat{v}}$, giving each weight its own effective learning rate:
$$W = W - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$ <br>

Here, weights with large and consistent gradients take smaller steps whereas weights with small and infrequent gradients take larger steps. This leads to faster and more stable convergence than SGD, especially on datasets with varied feature scales like NSL-KDD.

## 🛡️Regularization 
During training, each neuron's output was ranomly zeroed out with probability `rate`. The surviving outputs are scaled up by $\frac{1}{1 - \text{rate}}$ to keep the expected sum the same. <br>
This forces the network not to rely on any single neuron, since any neuron might be absent on a given batch. It encourages more redundant, generalizable features and is an effective tool against overfitting (when models memorize the training data instead of learning from it). Dropout is disabled during testing so the full network is used for the acutal predictions.

## ✂️Gradient Clipping
Before each optimizer update, each layer's gradient norm is computed, and if it exceeds a threshold, the gradients are rescaled down proportionally:
```python
norm = sqrt(sum(dweights**2) + sum(dbiases**2))
if norm > max_norm:
    dweights *= max_norm / norm
    dbiases  *= max_norm / norm
```
This caps out how large a single parameter update can be. It was key for the NSL-KDD dataset since without it, the Adam optimizer's first update on the large input space caused the loss to spike and the network to collapse into predicting a single class for every input.

## ⚖️Class Imbalance
The NSL-KDD dataset's class sizes are unbalanced, with normal and dos traffic making up a alrge portion of the training data while r2l and u2r make up 1-2% of the training data each. The loss straegy used here, categorical cross-entropy, minimizes the average loss across all samples. Initially this caused the model to just guess every r2l or u2r example wrong, and because they make up a small percentage of the data, they wouldn't change the average loss much. The model then ended up with 0% accuracy for both the r2l and u2r classes. <br>
To combat this, class weighting was used. Each class was given an inverse-frequency weight: $weight_c = \frac{totalSamples}{numClasses*count_c}$ <br>
This formula gives classes with fewer examples proportionally larger weights (capped at a max of 20), and vice versa. The weight is applied to the loss and to the backpropagated gradient itself. <br>
The results were evident, with accuracies for r2l and u2r going from 0% for both to 34% and 33%, respectivly.



## 🔁Training Loop
The training is done in 20 epochs. Each epoch shuffles the training data and splits the training data into batches (256 for Fashion MNIST, 64 for NSL-KDD). Then, for each batch, we run the forward pass, compute loss and accuracy, run the backward pass, clip gradients, and then call the optimizer to update weights. 

## 🎯Results

### Fashion MNIST
```text
Epoch  1/20  loss: 1.3488  acc: 57.9% 
Epoch  2/20  loss: 0.7752  acc: 74.6% 
Epoch  3/20  loss: 0.6528  acc: 78.3% 
Epoch  4/20  loss: 0.5940  acc: 80.1% 
Epoch  5/20  loss: 0.5584  acc: 81.2% 
Epoch  6/20  loss: 0.5336  acc: 81.9% 
Epoch  7/20  loss: 0.5143  acc: 82.5% 
Epoch  8/20  loss: 0.4987  acc: 83.0% 
Epoch  9/20  loss: 0.4870  acc: 83.3% 
Epoch 10/20  loss: 0.4759  acc: 83.7% 
Epoch 11/20  loss: 0.4670  acc: 84.0% 
Epoch 12/20  loss: 0.4590  acc: 84.2% 
Epoch 13/20  loss: 0.4512  acc: 84.5%
Epoch 14/20  loss: 0.4451  acc: 84.7% 
Epoch 15/20  loss: 0.4394  acc: 84.9%
Epoch 16/20  loss: 0.4342  acc: 85.0% 
Epoch 17/20  loss: 0.4291  acc: 85.2% 
Epoch 18/20  loss: 0.4246  acc: 85.3%
Epoch 19/20  loss: 0.4204  acc: 85.4% 
Epoch 20/20  loss: 0.4163  acc: 85.6% 

── Test set evaluation ── <br>
Loss:     0.4536 <br>
Accuracy: 84.0% $$
```
The network starts with a 57.9% accuracy and ends at 85.6% after training, and a test accuracy of 84.0%. For reference, a random baseline for a 10 class data set would be ~10% accuracy.

### NSL-KDD
```text
Epoch  1/20  loss: 0.4595  acc: 80.2%
Epoch  2/20  loss: 0.2545  acc: 89.9%
Epoch  3/20  loss: 0.2362  acc: 91.0%
Epoch  4/20  loss: 0.2169  acc: 91.7%
Epoch  5/20  loss: 0.2027  acc: 92.2%
Epoch  6/20  loss: 0.2011  acc: 92.3%
Epoch  7/20  loss: 0.1870  acc: 92.6%
Epoch  8/20  loss: 0.1870  acc: 92.7%
Epoch  9/20  loss: 0.1820  acc: 92.8%
Epoch 10/20  loss: 0.1723  acc: 92.9%
Epoch 11/20  loss: 0.1689  acc: 93.1%
Epoch 12/20  loss: 0.1709  acc: 93.0%
Epoch 13/20  loss: 0.1649  acc: 93.2%
Epoch 14/20  loss: 0.1616  acc: 93.3%
Epoch 15/20  loss: 0.1611  acc: 93.4%
Epoch 16/20  loss: 0.1657  acc: 93.3%
Epoch 17/20  loss: 0.1574  acc: 93.3%
Epoch 18/20  loss: 0.1571  acc: 93.4%
Epoch 19/20  loss: 0.1629  acc: 93.3%
Epoch 20/20  loss: 0.1575  acc: 93.4%
 
── Test set evaluation ── <br>
Loss:     1.0790 <br>
Accuracy: 78.2%
```

Here, the network starts off at 80.2% accuracy and eventually reaches 93.4% training accuracy - much thanks to the addition of Leaky ReLU, Adam optimizer, gradient clipping, class weighting, and regularization. The test accuracy of 78.2% is a larger difference, but looking at the dataset gives us some insight as to why. r2l (remote-to-local exploit) can often look like normal activity when looking at features such as traffic colume, error rates, and connection counts - which are some of the features of this dataset. More interestingly, the u2r (user-to-root exploit) class only appears 67 times out of ~22,000 records in the training set. This low representation makes it hard for the monel to understand what characterizes u2r, and in turn hard for it to predict it - although the class weighting does make a good attempt to mitigate this. <br>


References: <br>
[https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) <br>
[https://towardsdatascience.com/kaiming-he-initialization-in-neural-networks-math-proof-73b9a0d845c4/](https://towardsdatascience.com/kaiming-he-initialization-in-neural-networks-math-proof-73b9a0d845c4/) <br>
[https://www.geeksforgeeks.org/deep-learning/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/](https://www.geeksforgeeks.org/deep-learning/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/) <br>
[https://www.geeksforgeeks.org/deep-learning/categorical-cross-entropy-in-multi-class-classification/](https://www.geeksforgeeks.org/deep-learning/categorical-cross-entropy-in-multi-class-classification/) <br>
[https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)<br>
[https://github.com/Sentdex/nnfs](https://github.com/Sentdex/nnfs)<br>
[https://github.com/jmnwong/NSL-KDD-Dataset](https://github.com/jmnwong/NSL-KDD-Dataset)<br>
[https://www.kaggle.com/datasets/hassan06/nslkdd](https://www.kaggle.com/datasets/hassan06/nslkdd)<br>
[https://medium.com/@LayanSA/complete-guide-to-adam-optimization-1e5f29532c3d](https://medium.com/@LayanSA/complete-guide-to-adam-optimization-1e5f29532c3d)<br>
