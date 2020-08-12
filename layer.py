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

#Sigmoid
class Sigmoid(Layer):
    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def d_activation(x):
        return Sigmoid.activation(x) * (1 - Sigmoid.activation(x))
