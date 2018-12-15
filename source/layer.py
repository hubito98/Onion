from neuron import *
import random


class Layer(object):

    def __init__(self, input_size, output_size, activation=None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.neurons = list()
        for i in range(output_size):
            self.neurons.append(Neuron(weights=[
                Node(random.random()) for j in range(input_size)
            ], bias=Node(random.random()), activation=self.activation))

    def forward(self, inputs):
        outputs = list()
        for i in range(self.output_size):
            outputs.append(self.neurons[i].forward(inputs))
        return outputs

    def backward(self):
        for i in range(self.output_size):
            self.neurons[i].backward()