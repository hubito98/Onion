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

for x, y in zip(train_data['feature'][:100], train_data['label'][:100]):
    train_feature.append(tools.Node(x))
    train_label.append(tools.Node(y))

# train neural network for 10 epochs
for i in range(10):
    for x, y in zip(train_feature, train_label):
        neural_network.forward([x])
        neural_network.backward([y])

# evaluate on unseen 100 examples
avg_abs_error = 0
for i in range(len(train_data['feature'][100:])):
    feature = [train_data['feature'][100 + i]]
    label = train_data['label'][100 + i]
    predict = neural_network.predict(feature)[0]
    avg_abs_error += abs(predict - label)

print ("Average error: {}".format(avg_abs_error/len(train_data['feature'][100:])))
