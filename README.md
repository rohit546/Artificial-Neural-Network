# Artificial-Neural-Network


Artificial Neural Networks
An artificial neural network (ANN) is a machine learning model inspired by the structure
and function of the human brain's interconnected network of neurons. It consists of
interconnected nodes called artificial neurons, organized into layers. Information flows
through the network, with each neuron processing input signals and producing an output
signal that influences other neurons in the network.


![image](https://github.com/rohit546/Artificial-Neural-Network/assets/100420859/aade2c63-6efd-4bdc-adba-f7b276e28091)

Perceptron
A perceptron is the smallest element of a neural network. Perceptron was introduced by
Frank Rosenblatt in 1957. He proposed a Perceptron learning rule based on the original
McCulloch–Pitts (MCP) neuron. A perceptron is a type of neural network algorithm for
supervised learning of binary classifiers. This algorithm enables neurons to learn and
processes elements in the training set one at a time.

![image](https://github.com/rohit546/Artificial-Neural-Network/assets/100420859/677fd0e1-a5f1-4639-97cc-d369bf88eca2)


Components of Perceptron
The Perceptron, a pivotal concept in machine learning, operates as an artificial
neural network. It may manifest as a single layer or a multilayer structure. Its
essential elements include:
1. Inputs: Constituting one or more neurons, the input layer receives signals from
the external environment or other neural network layers.
2. Weights: Corresponding to each input neuron, weights signify the connection
strength between input and output neurons.
3. Activation Function: This function calculates the perceptron's output based on
the weighted sum of inputs and a bias term. Common activation functions
include the step function, sigmoid function, and ReLU function.
4. Output: The perceptron yields a singular binary value, 0 or 1, denoting the
classification of input data.
The Perceptron algorithm adjusts input signal weights to delineate a linear decision
boundary.
Steps to Execute the Perceptron Learning Algorithm
1. Feed the features of the model that is required to be trained as input in the first
layer.

2. All weights and inputs will be multiplied – the multiplied result of each weight and
input will be added up
3. The Bias value will be added to shift the output function
4. This value will be presented to the activation function (the type of activation
function will depend on the need)
5. The value received after the last step is the output value.

Perceptron - emulating the behavior of a logical AND gate
Suppose that we are going to work on AND Gate problem. The gate returns if and only if
both inputs are true. Our goal is by using the perceptron formula to mimic the behavior of a
logical AND.


![image](https://github.com/rohit546/Artificial-Neural-Network/assets/100420859/7c64b32f-8e78-4f82-9ac0-1c116610acad)




We are going to set weights randomly. Let’s say that w1 = 0.9 and w2 = 0.9
Round 1
We will apply 1st instance to the perceptron. x1 = 0 and x2 = 0.



Sum unit will be 0 as calculated below
Σ = x1 * w1 + x2 * w2 = 0 * 0.9 + 0 * 0.9 = 0
Activation unit checks sum unit is greater than a threshold. If this rule is satisfied, then it is
fired and the unit will return 1, otherwise it will return 0. BTW, modern neural networks
architectures do not use this kind of a step function as activation.
Activation threshold would be 0.5.
Sum unit was 0 for the 1st instance. So, activation unit would return 0 because it is
less than 0.5. Similarly, its output should be 0 as well. We will not update weights
because there is no error in this case.
Let’s focus on the 2nd instance. x1 = 0 and x2 = 1.
Sum unit: Σ = x1 * w1 + x2 * w2 = 0 * 0.9 + 1 * 0.9 = 0.9
What about errors?
Activation unit will return 1 because sum unit is greater than 0.5. However, output of this
instance should be 0. This instance is not predicted correctly. That’s why, we will update
weights based on the error.
ε = actual – prediction = 0 – 1 = -1
We will add error times learning rate value to the weights. Learning rate would be 0.5. BTW,
we mostly set learning rate value between 0 and 1.
w1 = w1 + α * ε = 0.9 + 0.5 * (-1) = 0.9 – 0.5 = 0.4
w2 = w2 + α * ε = 0.9 + 0.5 * (-1) = 0.9 – 0.5 = 0.4
Focus on the 3rd instance. x1 = 1 and x2 = 0.
Sum unit: Σ = x1 * w1 + x2 * w2 = 1 * 0.4 + 0 * 0.4 = 0.4
Activation unit will return 0 this time because output of the sum unit is 0.5 and it is less
than 0.5. We will not update weights.
Mention the 4rd instance. x1 = 1 and x2 = 1.
Sum unit: Σ = x1 * w1 + x2 * w2 = 1 * 0.4 + 1 * 0.4 = 0.8


Activation unit will return 1 because output of the sum unit is 0.8 and it is greater than the
threshold value 0.5. Its actual value should 1 as well. This means that 4th instance is
predicted correctly. We will not update anything.
Round 2
In previous round, we’ve used previous weight values for the 1st instance and it was
classified correctly. Let’s apply feed forward for the new weight values.
Remember the 1st instance. x1 = 0 and x2 = 0.
Sum unit: Σ = x1 * w1 + x2 * w2 = 0 * 0.4 + 0 * 0.4 = 0.4
Activation unit will return 0 because sum unit is 0.4 and it is less than the threshold value
0.5. The output of the 1st instance should be 0 as well. This means that the instance is
classified correctly. We will not update weights.
Feed forward for the 2nd instance. x1 = 0 and x2 = 1.
Sum unit: Σ = x1 * w1 + x2 * w2 = 0 * 0.4 + 1 * 0.4 = 0.4
Activation unit will return 0 because sum unit is less than the threshold 0.5. Its output
should be 0 as well. This means that it is classified correctly and we will not update


![image](https://github.com/rohit546/Artificial-Neural-Network/assets/100420859/8e5d3d94-b865-44d5-8b1b-49c266b3965b)


We’ve applied feed forward calculation for 3rd and 4th instances already for the current
weight values in the previous round. They were classified correctly.

National University

O f C o m p u t e r & E m e r g i n g S c i e n c e s F a i s a l a b a d - C h i n i o t C a m p u s

Page
7

Learning
We should continue this procedure until learning completed. We can terminate the learning
procedure here. Luckily, we can find the best weights in 2 rounds.
Updating weights means learning in the perceptron. We set weights to 0.9 initially but it
causes some errors. Then, we update the weight values to 0.4. In this way, we can predict all
instances correctly.
Note: A single-layer perceptron used to solve linear problems, the AND and OR Gates that
are inherently linearly separable. However, when it comes to XOR Gate, a single-layer
perceptron falls short due to the non-linear separability of its behavior. This limitation led
to the evolution of the perceptron into the multilayer perceptron, enabling the solution of
non-linear problems.
Python - emulating the behavior of a logical AND gate:
![image](https://github.com/rohit546/Artificial-Neural-Network/assets/100420859/9aa2d560-c01b-41fb-adce-85f9c832a2fa)
![image](https://github.com/rohit546/Artificial-Neural-Network/assets/100420859/6653266c-816c-41b9-8519-964a6b8c9571)


Scikit-Learn’s Perceptron
Scikit-Learn provides a Perceptron class that can be used pretty much as you would expect -
for example, on the iris dataset. The Iris flower data set consists of 50 samples from each of
three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). Four features were
measured from each sample: the length and the width of the sepals and petals, in
centimeters. You can use a publicly available iris (flower data set) dataset, or you load the
iris dataset using scikit-learn.


![image](https://github.com/rohit546/Artificial-Neural-Network/assets/100420859/16987f06-4d56-4974-a987-63d493e6be30)



Task 1: Single layer Perceptron to emulate the behavior of a logical OR


Implement the perceptron learning process to emulate the behavior of a logical OR gate,
without using any library functions. The OR gate outputs 0 only when both inputs are 0
otherwise 1.


![image](https://github.com/rohit546/Artificial-Neural-Network/assets/100420859/40ccca41-6b73-496d-abfa-c59d87fec0c1)


i. Start with random weights at first iteration; after that, use the updated weights on each
iteration. The weights and the bias of a perceptron can be learnt/updated by using
Perceptron Learning Rule (PLR) or Delta Rule. The formula for updating the weights
using the Delta Rule is typically expressed as:
∆wij = η. (di − yi

). xij where:

∆wij is the change in weight between neuron i and neuron j
η is the learning rate (a small positive constant),
di
is the desired output (either 0 or 1),
yi
is the actual output (0 or 1), and
xij is the input from neuron j to neuron i.
ii. Perform experiments with step activation function, typically expressed as:
y = {
0, x < n
1, x ≥ n
x represents the input to the step function, and n represents threshold (i.e. n=0.5) is the
value at which the step function transitions from 0 to 1.
iii. Analyze the effect of the number of epochs on convergence and print the weights of
the perceptron after training for each number of epochs. But your epochs must be
different from your fellows.
iv. Perform exercises with three different learning rates of your own choice but yours
learning must be different from your fellows.


Task 2: Multi-layer Perceptron to emulate the behavior of a logical XOR
Implement the perceptron learning process to emulate the logical XOR behavior, without

using any library functions. Due to the non-linear separability of XOR behavior, a single-
layer perceptron cannot effectively implement XOR. To address this, adjust the initial

weights and learning rate values accordingly, incorporating a hidden layer and
backpropagation algorithm for XOR gate. The Perceptron Delta Rule is effective
algorithm for training single-layer perceptron to learn linearly separable patterns (see. Task
1). However, it has limitations and cannot handle problems that require non-linear decision
boundaries. For more complex tasks, multilayer perceptron and other advanced techniques
like backpropagation are used.
![image](https://github.com/rohit546/Artificial-Neural-Network/assets/100420859/1d8f4698-9b17-483e-98af-ea14461d9faa)


i. Perform experiments with the hyperparameters given below and Sigmoid
activation function, typically expressed as:
σ(x) =
1
1 + e
−x

x represents the input to the x represents the input to the sigmoid function, and the
function returns a value between 0 and 1.
Hyperparameters:
hidden_units = 2, refers to the number of neurons (or units) in the hidden layer(s) of the
network.
learning_rate = 0.1, Commonly used learning rates range from 10−6
to 0.1

epochs = 10000, refer to the number of times the entire dataset is passed forward and backward
through the neural network during the training process
Note: you can get aid for task 1 and task 2, Hands-On Machine Learning with Scikit-Learn, Keras, and
TensorFlow, 3rd Edition, Aurélien Géron, Chapter 10. Introduction to Artificial Neural Networks. or
MathWorks, “Perceptron Neural Networks”
https://www.mathworks.com/help/deeplearning/ug/perceptron-neural-networks.html


Task 3: Multi-layer fully connected neural network with one hidden layer

Implement a multi-layer fully connected neural network with one hidden layer for hand-
written digits recognition and display some example predictions. You can use a publicly

available MNIST ((hand-written digit image) dataset, or you load the MNIST dataset using
scikit-learn.

![image](https://github.com/rohit546/Artificial-Neural-Network/assets/100420859/511338e8-9ef2-4d61-9dfc-dbfbb070c322)



