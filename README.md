# primitive-perceptron
This repository contains a simple implementation of the perceptron algorithm in Python, a type of artificial neural network used for binary classification. This is my first attempt at implementing a perceptron based on my knowledge acquired in my first AI class. The implementation is well-documented and follows best practices for Python code.
The Perceptron class has methods for training the perceptron on a set of samples and expected outputs, classifying new samples, updating the weights and bias based on the error between the expected output and the actual output, printing the activation function in a human-readable format, and plotting the samples and the line equation of the latest activation function.

# Installation
To use the Perceptron class, you will need to have NumPy and Matplotlib installed. You can install these packages using pip or with the requirements.txt file:


```bash
pip install numpy matplotlib
```
or
```bash
pip install -r requirements.txt
```
# Usage
Here's an example of how to use the Perceptron class:

```python
import numpy as np
from perceptron import Perceptron

# Create a perceptron with 2 attributes
perceptron = Perceptron(num_attributes=2)

# Define the samples and expected outputs
samples = np.array([[-1.0, 3.0], [2.0, 1.0], [2.0, -1.0], [1.0, 1.0], [-2.0, -1.0], [1.5, 3.0]])
output = np.array([1, 0, 1, 0, 1, 0])

# Train the perceptron on the samples and expected outputs
perceptron.train(samples, output)

# Print the activation function in a human-readable format
perceptron.print_activation_function()

# Plot the samples and the line equation of the latest activation function
perceptron.plot_samples_and_activation_function(samples, output)
```
# Methods
The Perceptron class has the following methods:

+ __init__(self, num_attributes): Initializes the perceptron with random weights and a bias of -1.
+ activation_function(self, attributes): Calculates the activation value for a given set of attributes.
+ classify(self, attributes): Classifies the activation value as 0 or 1.
+ update_weights_and_bias(self, attributes, expected_output): Updates the weights and bias based on the error between the expected output and the actual output.
+ train(self, samples, output, max_epochs=100): Trains the perceptron on the given samples and expected outputs.
+ print_activation_function(self): Prints the activation function in a human-readable format.
+ plot_samples_and_activation_function(self, samples, output): Plots the given samples and the line equation of the latest activation function.

# Error Handling
The Perceptron class includes error handling to check for invalid input in several methods. If the input is invalid, the methods will raise a ValueError with an appropriate error message.

# Style Guide
The code follows the PEP 8 style guide for Python code, which includes using lowercase with underscores for function and variable names, using docstrings to document the class and its methods, and adding appropriate spacing and indentation to improve readability. The code also includes type hints to make the function signatures more clear and self-documenting.

# License
This project is licensed under the MIT License .
