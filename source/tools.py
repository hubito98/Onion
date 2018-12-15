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
        self.right_node = y
        self.output_node = Node(self.left_node.value * self.right_node.value)
        return self.output_node

    def backward(self):
        self.left_node.derivative = self.right_node.value * self.output_node.derivative
        self.right_node.derivative = self.left_node.value * self.output_node.derivative


class AddOperation(object):

    def __init__(self):
        self.input_nodes = None
        self.output_node = None

    def forward(self, inputs):
        self.input_nodes = inputs
        score = 0
        for node in self.input_nodes:
            score += node.value
        self.output_node = Node(score)
        return self.output_node

    def backward(self):
        for i in range(len(self.input_nodes)):
            self.input_nodes[i].derivative = 1 * self.output_node.derivative


class MeanSquaredError(object):

    def __init__(self):
        self.input_node = None
        self.correct_value = None
        self.output_node = None

    def forward(self, x, correct_value):
        self.input_node = x
        self.correct_value = correct_value
        self.output_node = Node((self.input_node.value - correct_value)**2)
        return self.output_node

    def backward(self):
        self.input_node.derivative = \
            2 * (self.input_node.value - self.correct_value) * self.output_node.derivative
