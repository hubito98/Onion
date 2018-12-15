from tools import *


class Neuron(object):

    def __init__(self, weights, bias):
        self.input_size = len(weights)
        self.weights = weights
        self.bias = bias
        self.mul_operations = list()
        self.mul_results = list()
        self.add_operation = AddOperation()
        self.add_result = None
        for i in range(self.input_size + 1):        # additional multiplication for bias (1 * bias)
            self.mul_operations.append(MultiplyOperation())
            self.mul_results.append(None)

        self.loss = MeanSquaredError()
        self.loss_result = None
        self.lr = 0.001

    def forward(self, inputs, f):
        for i in range(self.input_size):
            self.mul_results[i] = self.mul_operations[i].forward(inputs[i], self.weights[i])

        self.mul_results[self.input_size] = self.mul_operations[self.input_size].forward(Node(1.0), self.bias)
        self.add_result = self.add_operation.forward(self.mul_results)
        self.loss_result = self.loss.forward(self.add_result, f)
        return self.add_result, self.loss_result

    def backward(self):
        self.loss_result.derivative = 1
        self.loss.backward()
        self.add_operation.backward()
        for i in range(self.input_size + 1):
            self.mul_operations[i].backward()

        for i in range(self.input_size):
            current_weight = self.weights[i]
            current_weight.value -= current_weight.derivative * self.lr
        self.bias.value -= self.bias.derivative * self.lr
