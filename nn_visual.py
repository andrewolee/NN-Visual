import numpy as np
import matplotlib.pyplot as plt
from layer import Relu, Sigmoid
from sequential import Sequential

training_set_size = 5000
epochs = 9
learning_rate = 0.01

model = Sequential([
    Relu((5, 2)),
    Sigmoid((1, 5))
])

delta = 0.02
Y, X = np.mgrid[slice(-1, 1 + delta, delta), slice(-1, 1 + delta, delta)]

cmap = plt.get_cmap('Reds')

for epoch in range(epochs):
    data = np.random.standard_normal((training_set_size, 2))
    labels = [1 if np.linalg.norm(x) < 0.5 else 0 for x in data]
    model.train(data, labels, learning_rate)

    plt.subplot(3, 3, epoch + 1)
    Z = np.reshape([model.predict(i) for i in zip(X.flatten(), Y.flatten())], X.shape)
    plt.contourf(X, Y, Z, cmap=cmap)

plt.show()

