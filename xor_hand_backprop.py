import numpy as np
import matplotlib
matplotlib.use(backend="QtAgg")
import matplotlib.pyplot as plt


# loss function (MSE)

def mse(targets, predictions):
    return (np.sum((targets - predictions), axis=1)**2)/predictions.shape[0]

def sigmoid(X):
    return 1/(1+np.exp(-X))

def sig_grad(X):
    return sigmoid(X)*(1-sigmoid(X))
# create the data
data = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        ])
# seperate the training data and targets
X = data[:, 0:2]
X.shape
X = X.T
X.shape
Y = data[:, -1]
Y = Y[None,...]
Y.shape
# intialize the input weights + bias of the model
no_hidden_neurons = 2
no_out = 1
W1 = np.random.uniform(size=(no_hidden_neurons,X.shape[0]))
B1 = np.random.uniform(size=(no_hidden_neurons,1))

# initialize the hidden weights + bias of the model
W2 = np.random.uniform(size=(no_out, no_hidden_neurons))
B2 = np.random.uniform(size=(no_out,))

for epoch in range(10000):

    # forward pass
    h = W1@X + B1
    h_out = sigmoid(h)
    out = W2@h_out + B2
    predictions = sigmoid(out)
    mse_loss = mse(Y, predictions)
    print(mse_loss)

    # backwards pass
    dpredictions = -2*(Y-predictions)/predictions.shape[1]
    dout = sig_grad(out)*dpredictions

    dW2 = dout@h_out.T
    dh_out = W2.T@dout
    dB2 = dout.sum(1)

    dh = sig_grad(h)*dh_out
    dW1 = dh@X.T
    dB1 = dh.sum(1).reshape(no_hidden_neurons,1)

    # apply the gradients
    lr = 0.3 
    B2 -= lr*dB2
    W2 -= lr*dW2

    W1 -= lr*dW1
    B1 -=lr*dB1
# forward pass
#X = np.array([[0, 0]]).T
h = W1@X + B1
h_out = sigmoid(h)
out = W2@h_out + B2
predictions = sigmoid(out)
print(X)
print(predictions)
print(predictions>0.50)
print("W1: ", W1)
print("B1: ", B1)
print("W2: ", W2)
print("B2: ", B2)

# testing first example with manual multiplication
print(X[:,0])
n11 = sigmoid(X[0,0]*W1[0,0] + X[1,0]*W1[0,1] + B1[0])
n12 = sigmoid(X[0,0]*W1[1,0] + X[1,0]*W1[1,1] + B1[1])
sigmoid(n11*W2[0, 0]+n12*W2[0, 1] + B2).item()

