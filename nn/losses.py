import numpy as np


class Loss:
    def regularization_loss(self):

        regularization_loss = 0

        for layer in self.trainable_layers:

            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                    np.sum(np.abs(layer.weights))

            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                    np.sum(np.abs(layer.biases))

            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                    np.sum(layer.weights**2)

            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                    np.sum(layer.biases**2)

        return regularization_loss

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, *, include_regularization=False):
        data_loss = self.accumulated_sum / self.accumulated_count
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    # Set/remember trainable layers


class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs

        # Normalizing the gradient
        self.dinputs = self.dinputs / samples


class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true - dvalues) / outputs

        # normalize gradient
        self.dinputs = self.dinputs / samples


class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped)) + \
            (1-y_true) * np.log(1-y_pred_clipped)

        return np.mean(sample_losses, axis=-1)

    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = -(y_true / clipped_dvalues -
                         (1-y_true) / (1-clipped_dvalues)) / outputs

        # Normalize the gradient
        self.dinputs = self.dinputs / samples


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        ''' to overcome the log(0)->inf tha will cause a problem when averaging
            we replace each number less than 1e-7
            with 1e-7 and If any value in y_pred is greater than 
            1 - 1e-7, it will be set to 1 - 1e-7.'''
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # handle y_pred if it's scalar. example: [0, 1, 0, 1, 1]
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # one-hot encoding case
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.log(correct_confidences)

    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues

        # Normalization of the gradient
        self.dinputs = self.dinputs / samples
