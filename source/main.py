import pandas as pd
import matplotlib.pyplot as plt
import random
from neuron import *


train_data = pd.read_csv("../data/example_data.csv")

input_neuron1 = Neuron([Node(random.random())], Node(random.random()))
input_neuron2 = Neuron([Node(random.random())], Node(random.random()))

hidden_neuron1 = Neuron(weights=[
    Node(random.random()), Node(random.random())
], bias=Node(random.random()))

hidden_neuron2 = Neuron(weights=[
    Node(random.random()), Node(random.random())
], bias=Node(random.random()))

output_neuron = Neuron(weights=[
    Node(random.random()), Node(random.random())
], bias=Node(random.random()))


train_feature = list()
train_label = list()

for x, y in zip(train_data['feature'], train_data['label']):
    train_feature.append(Node(x))
    train_label.append(Node(y))


#create MeanSquaredError object
mse = MeanSquaredError()

for i in range(10):
    avg_loss = 0
    for x, y in zip(train_feature, train_label):
        # predict = neuron.forward([x])
        # loss = mse.forward(predict, y.value)
        # loss.derivative = 1
        # mse.backward()
        # neuron.backward()
        # avg_loss += loss.value
        i1_predict = input_neuron1.forward([x])
        i2_predict = input_neuron2.forward([x])
        h1_predict = hidden_neuron1.forward([i1_predict, i2_predict])
        h2_predict = hidden_neuron2.forward([i1_predict, i2_predict])
        output_predict = output_neuron.forward([h1_predict, h2_predict])
        loss = mse.forward(output_predict, y.value)
        loss.derivative = 1
        mse.backward()
        output_neuron.backward()
        hidden_neuron1.backward()
        hidden_neuron2.backward()
        input_neuron2.backward()
        input_neuron1.backward()
        avg_loss += loss.value
    print(avg_loss / len(train_label))