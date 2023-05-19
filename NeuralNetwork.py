import numpy as np

class NeuralNetwork:

    def __init__(self, input_layer, output_layer, hidden_layers=100, num_of_nodes=10, activation_function='sigmoid',
                 learning_rate=0.001, iterations_number=200):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.num_of_nodes = num_of_nodes

        self.weights = []
        self.biases = []
        self.activations = []

        curr_layer = self.input_layer
        for i in range(self.hidden_layers):
            self.weights.append(np.random.randn(curr_layer, self.num_of_nodes))
            self.biases.append(np.random.randn(self.num_of_nodes))
            self.activations.append(np.zeros(self.num_of_nodes))
            curr_layer = self.num_of_nodes

        self.weights.append(np.random.randn(curr_layer, self.output_layer))
        self.biases.append(np.random.randn(self.output_layer))
        self.activations.append(np.zeros(self.output_layer))

        if activation_function == 'sigmoid':
            self.activation_func = lambda x: 1 / (1 + np.exp(-x))
            self.activation_derivative = lambda x: x * (1 - x)
        elif activation_function == 'relu':
            self.activation_func = lambda x: np.maximum(0, x)
            self.activation_derivative = lambda x: np.where(x > 0, 1, 0)

        self.learning_rate = learning_rate
        self.iterations_number = iterations_number
    def propagate_forward(self, X):
        self.activations[0] = self.activation_func(np.dot(X, self.weights[0]) + self.biases[0])
        for i in range(1, self.hidden_layers + 1):
            self.activations[i] = self.activation_func(
                np.dot(self.activations[i - 1], self.weights[i]) + self.biases[i])
        return self.activations[-1]

    def propagate_backward(self, X, y_expected, y_received):
        error = y_expected - y_received
        delta = error * self.activation_derivative(y_received)
        for i in range(self.hidden_layers, -1, -1):
            delta = np.dot(delta, self.weights[i + 1].T) * self.activation_derivative(self.activations[i])
            self.weights[i] += np.dot(self.activations[i - 1].T, delta) * self.learning_rate
            self.biases[i] += np.sum(delta, axis=0) * self.learning_rate
        self.weights[0] += np.dot(X.T, delta) * self.learning_rate
        self.biases[0] += np.sum(delta, axis=0) * self.learning_rate

    def fit(self, X_train, y_train):
        for epoch in range(self.iterations_number):
            y_calculated = self.propagate_forward(X_train)
            self.propagate_backward(X_train, y_train, y_calculated)

    def predict(self, X_test):
        output = self.propagate_forward(X_test)
        return np.argmax(output, axis=1)

    def predict_prob(self, X_test):
        output = self.propagate_forward(X_test)
        return self.activation_func(output)

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy
