import tools


class Network(object):

    def __init__(self, layers, loss_function, optimizer):
        self.layers = layers
        self.loss_function = loss_function()
        self.optimizer = optimizer(self.layers)
        self.output = None

    # features_list -> list of arrays of features (as numbers)
    # labels_list -> list of arrays of labels (as numbers)
    def evaluate(self, features_list, labels_list):
        mean_squared_error = [0] * len(labels_list[0])
        for feature, label in zip(features_list, labels_list):
            predict = self.predict(feature)

            for j in range(len(mean_squared_error)):
                mean_squared_error[j] += 1.0 / len(features_list) * (predict[j] - label[j])**2

        print ("Mean squared error: {}".format(mean_squared_error))

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
    def fit(self, features_list, labels_list, epochs=1, learning_rate=0.01):
        train_features_nodes = list()
        train_label_nodes = list()

        for features, labels, in zip(features_list, labels_list):
            features_nodes_array = list()
            labels_nodes_array = list()
            for feature in features:
                features_nodes_array.append(tools.Node(feature))
            for label in labels:
                labels_nodes_array.append(tools.Node(label))
            train_features_nodes.append(features_nodes_array)
            train_label_nodes.append(labels_nodes_array)

        for epoch_no in range(epochs):
            avg_squared_loss = [0] * len(labels_list[0])
            train_features_number = len(train_features_nodes)
            for j in range(train_features_number):
                self.forward(train_features_nodes[j])
                loss = self.backward(train_label_nodes[j], learning_rate)
                for k, loss_column in enumerate(loss):
                    avg_squared_loss[k] += 1.0/train_features_number * loss_column.value
            print("Epoch {}/{}, loss: {}".format(epoch_no+1, epochs, avg_squared_loss))

    # features -> array of features (as Nodes) for one sample
    # output -> array of predicts (as Nodes) for one sample
    def forward(self, features):
        self.output = features
        for layer in self.layers:
            self.output = layer.forward(self.output)
        return self.output

    # labels -> array of labels (as Nodes) for one sample
    def backward(self, labels, learning_rate):
        loss = self.loss_function.forward(self.output, labels)
        for loss_column in loss:
            loss_column.derivative = 1
        self.loss_function.backward()
        for layer in reversed(self.layers):
            # compute gradient
            layer.backward()
        # apply changes
        self.optimizer.update_parameters(learning_rate)
        return loss
