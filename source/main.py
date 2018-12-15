import pandas as pd
import matplotlib.pyplot as plt
import random
from neuron import *

train_data = pd.read_csv("../data/example_data.csv")

neuron = Neuron([Node(random.random())], Node(random.random()))

train_feature = list()
train_label = list()

for x, y in zip(train_data['feature'], train_data['label']):
    train_feature.append(Node(x))
    train_label.append(Node(y))

for i in range(10):
    avg_loss = 0
    for x, y in zip(train_feature, train_label):
        _, loss = neuron.forward([x], y.value)
        neuron.backward()
        avg_loss += loss.value
    print(avg_loss / len(train_label))