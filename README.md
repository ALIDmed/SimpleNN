# Neural Networks From Scratch

# About

This is a neural network implementation completely from scratch using only Numpy. I started with a lovely book called "Neural Networks From Scratch" by Harrison Kinsley & Daniel Kukieła. I recommend it to everyone wants to get a deeper understanding of what happening under the hood of neural Networks

## The project covers the following concepts

- layer
- Activation Functions
  - ReLU
  - Sigmoid
  - Softmax
  - Linear
- Loss Functions
  - Categorical Cross-Entropy Loss
  - Binary Cross-Entropy Loss
  - Mean Absolute Loss
  - Mean Squared Loss
- Optimizers
  - SGD
  - Adagrad
  - RMSProp
  - Adam
- Dropout
- L1 and L2 regularization
- Backpropagagtion
- MNIST Dataset
- Model class wrapper
- save/load parameters
- save/load model

## Implementaion

### Data preparation

```python
from nn.data import create_data_mnist
import numpy as np

X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# shuffling
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# preprocessing/scaling
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5
```

### Model Creation

```python
import numpy as np
from nn.layer import Layer_Dense
from nn.model import Model
from nn.activations import Activation_ReLU, Activation_Softmax
from nn.losses import Loss_CategoricalCrossentropy
from nn.optimizers import Optimizer_Adam
from nn.accuracy import Accuracy_Categorical

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
model.train(X, y, validation_data=(X_test, y_test),
            epochs=10, batch_size=128, print_every=100)

# run prediction
confidences = model.predict(X_test[:5])
predictions = model.output_layer_activation.predictions(confidences)
print(predictions)

# saving the model
model.save('fashion_mnist.model')
```
