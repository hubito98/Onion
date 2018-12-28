import pandas as pd
import tools
import layer
import nn

# prepare data for train and test
train_data = pd.read_csv("../data/train.csv")

data_stats = train_data.describe().transpose()

# train data preprocessing
train_data['rm'] = (train_data['rm'] - data_stats['min']['rm']) / (data_stats['max']['rm'] - data_stats['min']['rm'])
train_data['lstat'] = (train_data['lstat'] - data_stats['min']['lstat']) / (data_stats['max']['lstat'] - data_stats['min']['lstat'])
train_data['crim'] = (train_data['crim'] - data_stats['min']['crim']) / (data_stats['max']['crim'] - data_stats['min']['crim'])
train_data['zn'] = (train_data['zn'] - data_stats['min']['zn']) / (data_stats['max']['zn'] - data_stats['min']['zn'])
train_data['indus'] = (train_data['indus'] - data_stats['min']['indus']) / (data_stats['max']['indus'] - data_stats['min']['indus'])
train_data['chas'] = (train_data['chas'] - data_stats['min']['chas']) / (data_stats['max']['chas'] - data_stats['min']['chas'])
train_data['nox'] = (train_data['nox'] - data_stats['min']['nox']) / (data_stats['max']['nox'] - data_stats['min']['nox'])
train_data['age'] = (train_data['age'] - data_stats['min']['age']) / (data_stats['max']['age'] - data_stats['min']['age'])
train_data['dis'] = (train_data['dis'] - data_stats['min']['dis']) / (data_stats['max']['dis'] - data_stats['min']['dis'])
train_data['rad'] = (train_data['rad'] - data_stats['min']['rad']) / (data_stats['max']['rad'] - data_stats['min']['rad'])
train_data['tax'] = (train_data['tax'] - data_stats['min']['tax']) / (data_stats['max']['tax'] - data_stats['min']['tax'])
train_data['ptratio'] = (train_data['ptratio'] - data_stats['min']['ptratio']) / (data_stats['max']['ptratio'] - data_stats['min']['ptratio'])
train_data['black'] = (train_data['black'] - data_stats['min']['black']) / (data_stats['max']['black'] - data_stats['min']['black'])

# create neural network with two layers
neural_network = nn.Network([
    layer.Layer(13, 8, activation=tools.Relu),
    layer.Layer(8, 1)
], loss_function=tools.MeanSquaredError, optimizer=tools.Momentum)


# prepare list of nodes with data for training
train_feature = list()
train_label = list()

for x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, y in zip(train_data['rm'],
                   train_data['lstat'],
                   train_data['crim'],
                   train_data['zn'],
                   train_data['indus'],
                   train_data['chas'],
                   train_data['nox'],
                   train_data['age'],
                   train_data['dis'],
                   train_data['rad'],
                   train_data['tax'],
                   train_data['ptratio'],
                   train_data['black'],
                   train_data['medv']):
    # it has to be arrays of arrays, because it's possible to have
    # more then one feature per sample
    train_feature.append([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13])
    # same with labels
    train_label.append([y])

# train network for 500 epochs
neural_network.fit(train_feature[:300], train_label[:300], epochs=500, learning_rate=0.00001)
# evaluate it on unseen data
neural_network.evaluate(train_feature[300:], train_label[300:])
