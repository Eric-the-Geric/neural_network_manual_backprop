import numpy as np
# todo to make it look nicer and more general
def mse(targets, predictions):
    return (np.sum((targets - predictions), axis=1)**2)/predictions.shape[0]

class Sigmoid:
    def __call__(self, X):
        return 1/(1+np.exp(-X))
    def grad(self, X):
        return self(X)*(1-self(X))

class Layer:
    def __init__(self, in_shape, n_neurons, grad=True, bias=True):
        self.in_shape = in_shape 
        self.n_neurons = n_neurons
        self.shape = (in_shape, n_neurons)
        self.w = np.random.uniform(size=(in_shape, n_neurons))
        if bias:
            self.b = np.random.uniform(1,n_neurons)

    def __call__(self, X):
        return self.w @ X + self.b

    def params(self):
        return [self.w] + [self.b]

class MLP:
    def __init__(self):
        self.layers = [Layer(2, 2), Sigmoid(), Layer(1, 2), Sigmoid()]
    def __call__(self, X):
        for l in self.layers:
            X=l(X)
        return X




data = np.array([[1, 0, 1, 0],
                 [1, 1, 0, 0]])
targets = [[0, 1, 1, 0]]
predictions = network = MLP()
Y = network(data)
mse(targets, Y)



