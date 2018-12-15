import tools

class Network(object):

    def __init__(self, layers, loss_function=None):
        self.layers = layers
        if loss_function is None:
            self.loss_function = tools.MeanSquaredError()
        else:
            self.loss_function = loss_function()
        self.output = None

    # features -> array of features (as numbers) for one sample
    # predict_list -> array of predicts (as numbert) for one sample
    def predict(self, features):
        features_nodes = list()
        for feature in features:
            features_nodes.append(tools.Node(feature))
        predict_nodes = self.forward(features_nodes)
        predict_list = list()
        for predict_column in predict_nodes:
            predict_list.append(predict_column.value)
        return predict_list

    # features -> array of features (as Nodes) for one sample
    # output -> array of predicts (as Nodes) for one sample
    def forward(self, features):
        self.output = features
        for layer in self.layers:
            self.output = layer.forward(self.output)
        return self.output

    # labels -> array of labels (as Nodes) for one sample
    def backward(self, labels):
        loss = self.loss_function.forward(self.output, labels)
        for loss_column in loss:
            loss_column.derivative = 1
        self.loss_function.backward()
        for layer in reversed(self.layers):
            layer.backward()
