import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """A simple implementation of the perceptron algorithm."""

    def __init__(self, num_attributes):
        """Initialize the perceptron with random weights and a bias of -1."""
        if num_attributes <= 0:
            raise ValueError("Number of attributes must be greater than zero.")
        self.weights = np.ones(num_attributes)
        self.bias = -1.0

    def activation_function(self, attributes):
        """Calculate the activation value for a given set of attributes."""
        if len(attributes) != len(self.weights):
            raise ValueError("Number of attributes must match the number of weights.")
        return np.dot(self.weights, attributes) + self.bias

    def classify(self, attributes):
        """Classify the activation value as 0 or 1."""
        return 1 if self.activation_function(attributes) >= 0 else 0

    def update_weights_and_bias(self, attributes, expected_output):
        """Update the weights and bias based on the error between the expected output and the actual output."""
        if len(attributes) != len(self.weights):
            raise ValueError("Number of attributes must match the number of weights.")
        if expected_output not in [0, 1]:
            raise ValueError("Expected output must be 0 or 1.")
        t = 1 if expected_output else -1
        self.bias += t
        self.weights += t * attributes

    def train(self, samples, output, max_epochs=100):
        """Train the perceptron on the given samples and expected outputs."""
        if len(samples) != len(output):
            raise ValueError("Number of samples must match the number of output labels.")
        if len(samples[0]) != len(self.weights):
            raise ValueError("Number of attributes in samples must match the number of weights.")
        if max_epochs <= 0:
            raise ValueError("Maximum number of epochs must be greater than zero.")
        epochs = 0
        correct_classifications = 0

        while correct_classifications < len(samples) and epochs < max_epochs:
            epochs += 1
            for x, y in zip(samples, output):
                if self.classify(x) != y:
                    correct_classifications = 0
                    self.update_weights_and_bias(x, y)
                else:
                    correct_classifications += 1

        print(f"Number of Epochs: {epochs}")
        print(f"Loops without Error: {correct_classifications}")
        print(f"Weights: {self.weights}")

    def print_activation_function(self):
        """Print the activation function in a human-readable format."""
        terms = []
        for i, weight in enumerate(self.weights):
            sign = "+" if weight >= 0 else "-"
            terms.append(f"{sign} {abs(weight)}x{i+1}")
        sign = "+" if self.bias >= 0 else "-"
        terms.append(f"{sign} {abs(self.bias)}")
        equation = " ".join(terms)
        print(f"Activation function found: {equation} = 0")

    def plot_samples_and_activation_function(self, samples, output):
        """Plot the given samples and the line equation of the latest activation function."""
        if len(samples) != len(output):
            raise ValueError("Number of samples must match the number of output labels.")
        if len(samples[0]) != len(self.weights):
            raise ValueError("Number of attributes in samples must match the number of weights.")
        # Plot the samples
        plt.scatter(samples[output == 0][:, 0], samples[output == 0][:, 1], marker='o', label='Class 0')
        plt.scatter(samples[output == 1][:, 0], samples[output == 1][:, 1], marker='x', label='Class 1')

        # Plot the activation function
        x = np.linspace(np.min(samples[:, 0]), np.max(samples[:, 0]), 2)
        y = (-self.bias - self.weights[0] * x) / self.weights[1]
        plt.plot(x, y, label='Activation function line Eq')

        plt.legend()
        plt.show()

# usage:
perceptron = Perceptron(num_attributes=2)
samples = np.array([[-1.0, 3.0], [2.0, 1.0], [2.0, -1.0], [1.0, 1.0], [-2.0, -1.0], [1.5, 3.0]])
output = np.array([1, 0, 1, 0, 1, 0])
perceptron.train(samples, output)
perceptron.print_activation_function()
perceptron.plot_samples_and_activation_function(samples=samples,output=output)

