import os
import numpy as np
from mnist import MNIST

mnist_loader = MNIST(os.path.join(os.getcwd(), 'data'))

X_train, y_train = mnist_loader.load_training()
X_test, y_test = mnist_loader.load_testing()

X_train = np.array([np.array(row, dtype=np.float32) / 255.0 for row in X_train])
X_test = np.array([np.array(row, dtype=np.float32) / 255.0 for row in X_test])

y_train = np.array(y_train, dtype=np.int32)
y_test = np.array(y_test, dtype=np.int32)

labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
num_classes = 10

y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

def relu(x):
    return np.maximum(x, 0)

def relu_grad(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    shift_x = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(shift_x)
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss(y_true, logits):
    m = y_true.shape[0]
    p = softmax(logits)
    log_likelihood = -np.log(p[range(m), np.argmax(y_true, axis=1)] + 1e-8)
    loss = np.sum(log_likelihood) / m
    return loss

def cross_entropy_loss_grad(y_true, logits):
    p = softmax(logits)
    grad = p - y_true
    return grad / y_true.shape[0]

INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 80
OUTPUT_SIZE = 10

np.random.seed(0)


class NNModel:

    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None

    def forward(self, X):
        X = (X - np.mean(X, axis=0, keepdims=True)) / (np.std(X, axis=0, keepdims=True) + 1e-8)

        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)

        return self.Z2, self.A2

    def backward(self, X, y, logits, learning_rate=0.01, clip_value=1.0):
        X = (X - np.mean(X, axis=0, keepdims=True)) / (np.std(X, axis=0, keepdims=True) + 1e-8)

        loss_grad = cross_entropy_loss_grad(y, logits)

        dZ2 = loss_grad
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_grad(self.A1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        dW1 = np.clip(dW1, -clip_value, clip_value)
        db1 = np.clip(db1, -clip_value, clip_value)
        dW2 = np.clip(dW2, -clip_value, clip_value)
        db2 = np.clip(db2, -clip_value, clip_value)

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2


model = NNModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

def compute_accuracy(logits, y_true):
    predicted_classes = np.argmax(logits, axis=1)
    true_classes = np.argmax(y_true, axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    return accuracy

EPOCHS = 1000
for epoch in range(EPOCHS):
    train_logits, y_pred = model.forward(X_train)
    model.backward(X_train, y_train, train_logits, learning_rate=0.1)

    if epoch % 100 == 0:
        loss = cross_entropy_loss(y_train, train_logits)
        test_logits, _ = model.forward(X_test)

        train_accuracy = compute_accuracy(train_logits, y_train)
        test_accuracy = compute_accuracy(test_logits, y_test)

        print(f"Epoch {epoch}, Loss: {loss}, Train Acc: {train_accuracy:.2f}, Test Acc: {test_accuracy:.2f}")
