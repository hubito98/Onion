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
