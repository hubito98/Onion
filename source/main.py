import pandas as pd
import tools
import layer
import nn

# prepare data for train
train_data = pd.read_csv("../data/example_data.csv")

# create neural network with one leyer
neural_network = nn.Network([
    layer.Layer(1, 1)
], loss_function=tools.MeanSquaredError)


# prepare list of nodes with data for training
train_feature = list()
train_label = list()

for x, y in zip(train_data['feature'], train_data['label']):
    # it has to be arrays of arrays, because it's possible to have
    # more then one feature per sample
    train_feature.append([x])
    # same with labels
    train_label.append([y])

# train network for 10 epochs
neural_network.fit(train_feature[:180], train_label[:180], epochs=10)
# evaluate it on unseen data
neural_network.evaluate(train_feature[180:], train_label[180:])
