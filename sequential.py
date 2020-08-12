import numpy as np
from layer import Relu, Sigmoid

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    @staticmethod
    def quad_cost(y, a):
        return np.square(y - a) / 2

    @staticmethod
    def d_quad_cost(y, a):
        return a - y

    def train(self, data, labels, learning_rate):
        for x, y in zip(data, labels):
            a = self.predict(x)
            gradient = self.d_quad_cost(y, a)
            for layer in reversed(self.layers):
                gradient = layer.back_prop(gradient)
            for layer in self.layers:
                layer.update(learning_rate)

    def predict(self, x):
        for layer in self.layers:
            x = layer.feed_foward(x)
        return x
