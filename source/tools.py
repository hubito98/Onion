import math


class Node(object):

    def __init__(self, value):
        self.value = value
        self.derivative = 0


class MultiplyOperation(object):

    def __init__(self):
        self.left_node = None
        self.right_node = None
        self.output_node = None

    def forward(self, x, y):
        self.left_node = x
        self.left_node.derivative = 0
        self.right_node = y
        self.right_node.derivative = 0
        self.output_node = Node(self.left_node.value * self.right_node.value)
        return self.output_node

    def backward(self):
        self.left_node.derivative += self.right_node.value * self.output_node.derivative
        self.right_node.derivative += self.left_node.value * self.output_node.derivative


class AddOperation(object):

    def __init__(self):
        self.input_nodes = None
        self.output_node = None

    def forward(self, inputs):
        self.input_nodes = inputs
        for i in range(len(self.input_nodes)):
            self.input_nodes[i].derivative = 0
        score = 0
        for node in self.input_nodes:
            score += node.value
        self.output_node = Node(score)
        return self.output_node

    def backward(self):
        for i in range(len(self.input_nodes)):
            self.input_nodes[i].derivative += 1 * self.output_node.derivative


class Sigmoid(object):

    def __init__(self):
        self.input_node = None
        self.output_node = None

    def forward(self, input_node):
        self.input_node = input_node
        self.input_node.derivative = 0
        self.output_node = Node(1/(1 + math.exp(-self.input_node.value)))
        return self.output_node

    def backward(self):
        self.input_node.derivative += \
            self.output_node.value * (1 - self.output_node.value) * self.output_node.derivative


class Relu(object):

    def __init__(self):
        self.input_node = None
        self.output_node = None

    def forward(self, input_node):
        self.input_node = input_node
        self.input_node.derivative = 0
        self.output_node = Node(max(0, self.input_node.value))
        return self.output_node

    def backward(self):
        if self.output_node.value > 0:
            self.input_node.derivative += 1 * self.output_node.derivative
        else:
            self.input_node.derivative += 0 * self.output_node.derivative


class Linear(object):

    def __init__(self):
        self.input_node = None
        self.output_node = None

    def forward(self, input_node):
        self.input_node = input_node
        self.input_node.derivative = 0
        self.output_node = Node(self.input_node.value)
        return self.output_node

    def backward(self):
        self.input_node.derivative += 1 * self.output_node.derivative


class MeanSquaredError(object):

    def __init__(self):
        self.predict = None
        self.correct_values = None
        self.output = None

    def forward(self, predict, label):
        self.predict = predict
        self.correct_values = list()
        self.output = list()
        for i in range(len(self.predict)):
            self.predict[i].derivative = 0
            self.correct_values.append(label[i].value)
            self.output.append(Node((self.predict[i].value - self.correct_values[i])**2))
        return self.output

    def backward(self):
        for i in range(len(self.predict)):
            self.predict[i].derivative += \
                2 * (self.predict[i].value - self.correct_values[i]) * self.output[i].derivative


class SGD(object):

    def __init__(self, layers):
        self.layers = layers

    def update_parameters(self, learning_rate):
        for layer in self.layers:
            for neuron in layer.neurons:
                for i in range(neuron.input_size):
                    current_weight = neuron.weights[i]
                    current_weight.value -= current_weight.derivative * learning_rate
                neuron.bias.value -= neuron.bias.derivative * learning_rate


class Momentum(object):

    def __init__(self, layers):
        self.layers = layers
        self.velocities = list()
        for i, layer in enumerate(layers):
            self.velocities.append(list())
            for j, neuron in enumerate(layer.neurons):
                self.velocities[i].append(list())
                for weight in neuron.weights:
                    self.velocities[i][j].append(0)
                self.velocities[i][j].append(0)     # one more then weights cause of bias

    def update_parameters(self, learning_rate):
        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer.neurons):
                for k in range(neuron.input_size):
                    current_weight = neuron.weights[k]
                    self.velocities[i][j][k] = 0.9 * self.velocities[i][j][k] +\
                                               current_weight.derivative * learning_rate
                    current_weight.value -= self.velocities[i][j][k]
                self.velocities[i][j][neuron.input_size] = 0.85 * self.velocities[i][j][neuron.input_size] + \
                                                           neuron.bias.derivative * learning_rate
                neuron.bias.value -= self.velocities[i][j][neuron.input_size]
