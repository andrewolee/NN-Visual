import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self, shape):
        self.shape = shape
        self.weights = np.random.standard_normal(shape)
        self.biases = np.random.standard_normal((shape[0], 1))
        
    @staticmethod
    @abstractmethod
    def activation(x):
        raise NotImplementedError("Activation function not defined")

    @staticmethod
    @abstractmethod
    def d_activation(x):
        raise NotImplementedError("Derivative of activation not defined")

    def feed_foward(self, x):
        x = np.reshape(x, (self.shape[1], 1))
        self.a = x
        self.z = self.weights@x + self.biases
        return self.activation(self.z)

    def back_prop(self, gradient):
        self.delta = gradient * self.d_activation(self.z)
        return self.weights.T@self.delta

    def update(self, learning_rate):
        self.weights -= learning_rate * (self.delta@self.a.T)
        self.biases -= learning_rate * self.delta


# Relu
class Relu(Layer):
    @staticmethod
    def activation(x):
        return np.where(x > 0, x, 0)

    @staticmethod
    def d_activation(x):
        return np.where(x > 0, 1, 0)

# Leaky Relu
class Lrelu(Layer):
    @staticmethod
    def activation(x):
        return np.where(x > 0, x, x * 0.1)

    @staticmethod
    def d_activation(x):
        return np.where(x > 0, 1, 0.1)

# Sigmoid
class Sigmoid(Layer):
    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def d_activation(x):
        return Sigmoid.activation(x) * (1 - Sigmoid.activation(x))

# Sine
class Sine(Layer):
    @staticmethod
    def activation(x):
        return np.sin(x)

    @staticmethod
    def d_activation(x):
        return np.cos(x)

# Augmented Sine
class Asine:
    def __init__(self, shape):
        self.shape = shape
        self.amp = np.ones((shape[0], 1))
        self.weights = np.random.standard_normal(shape)
        self.biases = np.random.standard_normal((shape[0], 1))

    @staticmethod
    def activation(x):
        return np.sin(x)

    @staticmethod
    def d_activation(x):
        return np.cos(x)

    def feed_foward(self, x):
        x = np.reshape(x, (self.shape[1], 1))
        self.a = x
        self.z = self.weights@x + self.biases
        self.act =  self.activation(self.z)
        return self.amp * self.act

    def back_prop(self, gradient):
        self.gradient = gradient
        self.delta = gradient * self.amp * self.d_activation(self.z)
        return self.weights.T@self.delta

    def update(self, learning_rate):
        self.amp -= learning_rate * self.gradient * self.act
        self.weights -= learning_rate * (self.delta@self.a.T)
        self.biases -= learning_rate * self.delta
