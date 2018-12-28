# Onion
Library for creating neural networks build from scratch.

## Use it
Just download files from source folder to your project.

## Create neural network
Like it is given in main.py you have to import nn and tools and then:

neural_network = nn.Network([
    layer.Layer(3, 8, activation=tools.Relu),
    layer.Layer(8, 1)
], loss_function=tools.MeanSquaredError, optimizer=tools.Momentum)

This example creating neural network with 3 inputs neurons, 8 hidden neurons and 1 output neuron.

Default activation function is linear.

## Train neural network
You have to give features as array of arrays, even if one sample feature is one number then features given to train
is for example [[1.0], [2.3], [1.1]], and labels in same format.
You can also give epochs number and learnign rate (by default epochs=1, learning_rate=0.01

neural_network.fit(train_feature, train_label, epochs=30, learning_rate=0.001)

Above we train our neural_network with train_feature and train_label arrays on 30 epochs with learning_rate=0.001

## Predict with neural network
You have to give array of features for one sample and you get array with predictions

neural_network.predict(feature)

Above line of code will give you prediction for feature as array of numbers.

## Evaluate neural network
You have to give features and labels (like in training) and mean squared error of all predictions will be printed.

neural_network.evaluate(test_feature, test_label)
