import numpy as np
import matplotlib.pyplot as plt
from layer import Relu, Sigmoid
from sequential import Sequential

training_set_size = 100000
epochs = 1
learning_rate = 0.08

model = Sequential([
    Relu((20, 2)),
    Sigmoid((1,20))
])

delta = 0.02
Y, X = np.mgrid[slice(-1, 1 + delta, delta), slice(-1, 1 + delta, delta)]

for epoch in range(epochs):
    data = np.random.standard_normal((training_set_size, 2))
    labels = [1 if np.linalg.norm(x) < 0.5 else 0 for x in data]
    model.train(data, labels, learning_rate)


Z = np.reshape([model.predict(i) for i in zip(X.flatten(), Y.flatten())], X.shape)

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title("nn visual")

plt.show()

