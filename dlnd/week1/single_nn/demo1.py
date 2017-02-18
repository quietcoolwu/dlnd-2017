from numpy import array, dot, exp, random


class NeuralNetwork(object):
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, iter_num):
        for each in range(iter_num):
            # pass the training set through our neural net
            output = self.predict(training_set_inputs)

            # calculate the error
            error = training_set_outputs - output

            # multiply the error by the input ad again by
            # the gradient of the sigmoid curve
            adj = dot(training_set_inputs.T,
                      error - self.__sigmoid_derivative(output))

            # adjust the weights
            self.synaptic_weights += adj

    def predict(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == '__main__':
    # initialize a single neuron network
    neural_network = NeuralNetwork()

    print('Random starting synaptic_weights:')
    print(neural_network.synaptic_weights)
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print('New synaptic_weights after training:')
    print(neural_network.synaptic_weights)

    print('Predicting:')
    print(neural_network.predict(array([[1, 0, 0]])))
