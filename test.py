import numpy as np
from nn.data import create_data_mnist
from nn.layer import Layer_Dense
from nn.model import Model
from nn.activations import Activation_ReLU, Activation_Softmax
from nn.losses import Loss_CategoricalCrossentropy
from nn.optimizers import Optimizer_Adam
from nn.accuracy import Accuracy_Categorical


# X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# # shuffling
# keys = np.array(range(X.shape[0]))
# np.random.shuffle(keys)
# X = X[keys]
# y = y[keys]

# # preprocessing/scaling
# X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
# X_test = (X_test.reshape(
#     X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

X = np.array([
    [1, 2, 2, 2],
    [2, 1, 2, 1]
])

y = np.array([1, 0])
model = Model()
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=1e-3, learning_rate=0.01),
    accuracy=Accuracy_Categorical()
)

model.finalize()
model.train(X, y, epochs=10, batch_size=128, print_every=100)

# save the model
model.save('fashion_mnist.model')
