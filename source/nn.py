import tools

class Network(object):

    def __init__(self, layers, loss_function=None):
        self.layers = layers
        if loss_function is None:
            self.loss_function = tools.MeanSquaredError()
        else:
            self.loss_function = loss_function()
        self.output = None

    # features_list -> list of arrays of features (as numbers)
    # labels_list -> list of arrays of labels (as numbers)
    def evaluate(self, features_list, labels_list):
        avg_abs_error = [0] * len(labels_list[0])
        for i in range(len(features_list)):
            feature = features_list[i]
            label = labels_list[i]
            predict = self.predict(feature)

            for j in range(len(avg_abs_error)):
                avg_abs_error[j] += 1.0 / len(features_list) * abs(predict[j] - label[j])

        print ("Average absolute error: {}".format(avg_abs_error))

    # features -> array of features (as numbers) for one sample
    # predict_list -> array of predicts (as numbers) for one sample
    def predict(self, features):
        features_nodes = list()
        for feature in features:
            features_nodes.append(tools.Node(feature))
        predict_nodes = self.forward(features_nodes)
        predict_list = list()
        for predict_column in predict_nodes:
            predict_list.append(predict_column.value)
        return predict_list

    # features_list -> list of arrays of features (as numbers)
    # labels_list -> list of arrays of labels (as numbers)
    def fit(self, features_list, labels_list, epochs=1):
        train_features_nodes = list()
        train_label_nodes = list()

        for x, y, in zip(features_list, labels_list):
            features_nodes_array = list()
            labels_nodes_array = list()
            for feature in x:
                features_nodes_array.append(tools.Node(feature))
            for label in y:
                labels_nodes_array.append(tools.Node(label))
            train_features_nodes.append(features_nodes_array)
            train_label_nodes.append(labels_nodes_array)

        for i in range(epochs):
            for j in range(len(train_features_nodes)):
                self.forward(train_features_nodes[j])
                self.backward(train_label_nodes[j])

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
