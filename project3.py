import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.utils import check_random_state


def accuracy(y_true, y_pred):
    return np.average(y_true==y_pred)


# Helper function to plot a decision boundary.
def plot_decision_boundary(pred_func, train_data, color):
    # Set min and max values and give it some padding
    x_min, x_max = train_data[:, 0].min() - .5, train_data[:, 0].max() + .5
    y_min, y_max = train_data[:, 1].min() - .5, train_data[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=color, cmap=plt.cm.RdYlGn)


class NeuralNetwork(object):
    def __init__(self, nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim, init="random"):
        """
        Descriptions:
            W1: First layer weights
            b1: First layer biases
            W2: Second layer weights
            b2: Second layer biases
            W3: Third layer weights
            b3: Third layer biases
        
        Args:
            nn_input_dim: (int) The dimension D of the input data.
            nn_hdim1: (int) The number of neurons  in the hidden layer H1.
            nn_hdim2: (int) The number of neurons H2 in the hidden layer H1.
            nn_output_dim: (int) The number of classes C.
            init: (str) initialization method used, {'random', 'constant'}
        
        Returns:
            
        """
        # reset seed before start
        np.random.seed(0)
        self.model = {}

        if init == "random":
            self.model['W1'] = np.random.randn(nn_input_dim, nn_hdim1)
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.random.randn(nn_hdim1, nn_hdim2)
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.random.randn(nn_hdim2, nn_output_dim)
            self.model['b3'] = np.zeros((1, nn_output_dim))

        elif init == "constant":
            self.model['W1'] = np.ones((nn_input_dim, nn_hdim1))
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.ones((nn_hdim1, nn_hdim2))
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.ones((nn_hdim2, nn_output_dim))
            self.model['b3'] = np.zeros((1, nn_output_dim))

    def forward_propagation(self, X):
        """
        Forward pass of the network to compute the hidden layer features and classification scores. 
        
        Args:
            X: Input data of shape (N, D)
            
        Returns:
            y_hat: (numpy array) Array of shape (N,) giving the classification scores for X
            cache: (dict) Values needed to compute gradients
            
        """
        W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
        
        ### CODE HERE ###
        h1 = X.dot(W1) + b1
        z1 = relu(h1)
        h2 = z1.dot(W2) + b2
        z2 = tanh(h2)
        h3 = z2.dot(W3) + b3
        
        y_hat = sigmoid(h3)

        
        cache = {}
        cache['h1'] = h1
        cache['h2'] = h2
        cache['h3'] = h3
        cache['z1'] = z1
        cache['z2'] = z2
        cache['y_hat'] = y_hat

        
        #################
        y_hat = y_hat.reshape(X.shape[0])
        assert y_hat.shape==(X.shape[0],), f"y_hat.shape is {y_hat.shape}. Reshape y_hat to {(X.shape[0],)}"
        cache = {'h1': h1, 'z1': z1, 'h2': h2, 'z2': z2, 'h3': h3, 'y_hat': y_hat}
    
        return y_hat, cache

    def back_propagation(self, cache, X, y, L2_norm=0.0):
        """
        Compute the gradients
        
        Args:
            cache: (dict) Values needed to compute gradients
            X: (numpy array) Input data of shape (N, D)
            y: (numpy array) Training labels (N, ) -> (N, 1)
            L2_norm: (int) L2 normalization coefficient
            
        Returns:
            grads: (dict) Dictionary mapping parameter names to gradients of model parameters
            
        """
        W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
        h1, z1, h2, z2, h3, y_hat = cache['h1'], cache['z1'], cache['h2'], cache['z2'], cache['h3'], cache['y_hat']
        
        # For matrix computation
        y = y.reshape(-1, 1)
        y_hat = y_hat.reshape(-1, 1)
        
        ### CODE HERE ###
        
        dy_hat = (-y)/(y_hat) + (1-y)/(1-y_hat)
        dh3 = dy_hat *(y_hat - y_hat**2)
        db3 = np.sum(dh3, axis = 0, keepdims = True)       
        dW3 = z2.T.dot(dh3) + (2* L2_norm * W3) 
        
        dz2 = dh3.dot(W3.T)        
        dh2 = dz2 * (1 - (z2**2))
        db2 = np.sum(dh2, axis = 0, keepdims = True)
        dW2 = z1.T.dot(dh2) + (2* L2_norm * W2)
        dz1 = dh2.dot(W2.T)

        
        dh1 = dz1 * (z1 > 0)
        dW1 = X.T.dot(dh1) + (2 * L2_norm * W1)
        db1 = np.sum(dh1, axis = 0, keepdims = True)
        
        #################
        
        grads = dict()
        grads['dy_hat'] = dy_hat
        grads['dh3'] = dh3
        grads['dW3'] = dW3
        grads['db3'] = db3
        grads['dW2'] = dW2
        grads['db2'] = db2
        grads['dW1'] = dW1
        grads['db1'] = db1

        return grads

    
    def compute_loss(self, y_pred, y_true, L2_norm=0.0):
        """
        Descriptions:
            Evaluate the total loss on the dataset
        
        Args:
            y_pred: (numpy array) Predicted target (N,)
            y_true: (numpy array) Array of training labels (N,)
        
        Returns:
            loss: (float) Loss (data loss and regularization loss) for training samples.
        """
        W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
        
        ### CODE HERE ###
        loss = (-1*y_true)*np.log(y_pred) - (1-y_true) * np.log(1-y_pred)
        total_loss = np.sum(loss) + (L2_norm * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)))
        #################

        return total_loss
        

    def train(self, X_train, y_train, X_val=None, y_val=None, learning_rate=1e-3, L2_norm=0.0, epoch=20000, print_loss=True):
        """
        Descriptions:
            Train the neural network using gradient descent.
        
        Args:
            X_train: (numpy array) training data (N, D)
            X_val: (numpy array) validation data (N, D)
            y_train: (numpy array) training labels (N,)
            y_val: (numpy array) valiation labels (N,)
            y_pred: (numpy array) Predicted target (N,)
            learning_rate: (float) Scalar giving learning rate for optimization
            L2_norm: (float) Scalar giving regularization strength.
            epoch: (int) Number of epoch to take
            print_loss: (bool) if true print loss during optimization

        Returns:
            A dictionary giving statistics about the training process
        """

        loss_history = []
        train_acc_history = []
        val_acc_history = []
        lr = learning_rate
        
        for it in range(epoch):
            ### CODE HERE ###       
            y_hat, cache = self.forward_propagation(X_train) # load y_hat, cache data from forward propagtion of X_train data
            loss = self.compute_loss(y_hat, y_train, L2_norm = L2_norm) # load loss function value from compute_loss function
            grads = self.back_propagation(cache, X_train, y_train, L2_norm = L2_norm) # load grads data from backpropagation function
            
            y_hat = y_hat - lr * grads['dy_hat']
            self.model['W1'] = self.model['W1'] - lr * grads['dW1']
            self.model['b1'] = self.model['b1'] - lr * grads['db1']
            self.model['W2'] = self.model['W2'] - lr * grads['dW2']
            self.model['b2'] = self.model['b2'] - lr * grads['db2']
            self.model['W3'] = self.model['W3'] - lr * grads['dW3']
            self.model['b3'] = self.model['b3'] - lr * grads['db3']
            
            ################# 
            if (it+1) % 1000 == 0:
                loss_history.append(loss)
                y_train_pred = self.predict(X_train)
                train_acc = np.average(y_train==y_train_pred)
                train_acc_history.append(train_acc)
                
                if X_val is not None:
                    y_val_pred = self.predict(X_val)
                    val_acc = np.average(y_val==y_val_pred)
                    val_acc_history.append(val_acc)

            if print_loss and (it+1) % 1000 == 0:
                print(f"Loss (epoch {it+1}): {loss}")
 
        if X_val is not None:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history,
            }
        else:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
            }

    def predict(self, X):
        ### CODE HERE ###
        y_pred = self.forward_propagation(X)[0] # determine y_hat with forward propagation function
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        return y_pred
        #################  



def tanh(x):
    ### CODE HERE ###
    x = np.tanh(x) # use existing tanh function in np.
    #################  
    return x
    

def relu(x):
    ### CODE HERE ###
    x = np.maximum(0,x) # using np.max and set 0 at negative part
    #################
    return x 


def sigmoid(x):
    ### CODE HERE ###
    #x = np.where(x>=0, 1/(1+np.exp(-x)), (np.exp(x))/(1+np.exp(x))) ## to prevent overflow at negative part, change to 1-sigmoid(-x) at negative.
    #x = 1/(1+np.exp(-x))
    x = np.where(x >= 0, 1/(1 + np.exp(-x)), np.exp(x)/(1 + np.exp(x)))
    
    #################
    return x


######################################################################################




class Linear(object):

    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for an linear layer.
        
        Args:
            x: (numpy array) Array containing input data, of shape (N, D)
            w: (numpy array) Array of weights, of shape (D, M)
            b: (numpy array) Array of biases, of shape (M,)

        Returns: 
            out: (numpy array) output, of shape (N, M)
            cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        out = x.dot(w) + b
        cache = (x, w, out)
        
        #################
        return out, cache

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for an linear layer.

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: (numpy array) Gradient with respect to x, of shape (N, D)
            dw: (numpy array) Gradient with respect to w, of shape (D, M)
            db: (numpy array) Gradient with respect to b, of shape (M,)
        """

        ### CODE HERE ###
        x, w, out = cache
        dx = dout.dot(w.T)
        dw = x.T.dot(dout)
        db = np.sum(dout, axis = 0)
        #################  
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified linear units (ReLUs).

        Args:
            x: (numpy array) Input

        Returns:
            out: (numpy array) Output
            cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        out = relu(x)
        cache = (x, out)
        
        #################  
        return out, cache

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs).

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        x, out = cache
        dx = dout * (x > 0 )
        #################  
        return dx

class Tanh(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of Tanh.

        Args:
            x: Input

        Returns:
            out: Output, array of the same shape as x
            cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        out = tanh(x)
        cache = (x, out)
        #################  
        return out, cache

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of Tanh.

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        x, out = cache
        dx = dout * (1 - (out**2))
        #################  
        return dx

class Sigmoid(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of Sigmoid.

        Args:
            x: Input

        Returns:
            out: Output
            cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        out = sigmoid(x)
        cache = (x, out)
        #################  
        return out, cache

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of Sigmoid.

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        x, out = cache
        dx = dout * (out - (out**2))
        #################  
        return dx


        
class SigmoidWithBCEloss(object): 

    @staticmethod
    def forward(x, y=None):
        """
        if y is None, computes the forward pass for a layer of sigmoid with binary cross-entropy loss.
        Else, computes the loss for binary classification.
        Args:
            x: Input data
            y: Training labels or None 
       
        Returns:
            if y is None:
                y_hat: (numpy array) Array of shape (N,) giving the classification scores for X
            else:
                loss: (float) data loss
                cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        
        y_hat = np.where(x >=0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)) )
        y_hat = y_hat.reshape(x.shape[0])
        
        if y is None : 
            return y_hat
        else : 
            loss = np.sum( (-y * np.log(y_hat)) - ((1-y_hat) * np.log(1-y_hat)))
            cache = (y_hat, y)
        ################# 
        
        assert y_hat.shape==(x.shape[0],), f"y_hat.shape is {y_hat.shape}. Reshape y_hat to {(x.shape[0],)}"
        return loss, cache

    @staticmethod
    def backward(cache, dout=None):
        """
        Computes the loss and gradient for softmax classification.
        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        y_hat, y = cache
        # For matrix computation
        y = y.reshape(-1, 1)
        y_hat = y_hat.reshape(-1, 1)
        
        ### CODE HERE ###
        if dout is None : 
            dx = y_hat - y
        else :
            dx = dout * (y_hat - y)
        ################# 
        return dx


class NeuralNetwork_module(object):
    def __init__(self, nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim, init="random"):
        """
        Descriptions:
            W1: First layer weights
            b1: First layer biases
            W2: Second layer weights
            b2: Second layer biases
            W3: Third layer weights
            b3: Third layer biases
        
        Args:
            nn_input_dim: (int) The dimension D of the input data.
            nn_hdim1: (int) The number of neurons  in the hidden layer H1.
            nn_hdim2: (int) The number of neurons H2 in the hidden layer H1.
            nn_output_dim: (int) The number of classes C.
            init: (str) initialization method used, {'random', 'constant'}
        
        Returns:
            
        """
        # reset seed before start
        np.random.seed(0)
        self.model = {}

        if init == "random":
            self.model['W1'] = np.random.randn(nn_input_dim, nn_hdim1)
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.random.randn(nn_hdim1, nn_hdim2)
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.random.randn(nn_hdim2, nn_output_dim)
            self.model['b3'] = np.zeros((1, nn_output_dim))

        elif init == "constant":
            self.model['W1'] = np.ones((nn_input_dim, nn_hdim1))
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.ones((nn_hdim1, nn_hdim2))
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.ones((nn_hdim2, nn_output_dim))
            self.model['b3'] = np.zeros((1, nn_output_dim))

    def forward(self, X, y=None):
        """
        Forward pass of the network to compute the hidden layer features and classification scores. 
        
        Args:
            X: Input data of shape (N, D)
            y: (numpy array) Training labels (N,) or None
            
        Returns:
            if y is None:
                y_hat: (numpy array) Array of shape (N,) giving the classification scores for X
            else:
                loss: (float) data loss
                cache: Values needed to compute gradients
            
        """

        W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
        cache = {}
        
        ### CODE HERE ###
        h1, cache['h1'] = Linear.forward(X, W1, b1)
        z1, cache['z1'] = ReLU.forward(h1)
                          
        h2, cache['h2'] = Linear.forward(z1, W2, b2)
        z2, cache['z2'] = Tanh.forward(h2)
                          
        out, cache['out'] = Linear.forward(z2, W3, b3) 
        
        #################  
        if y is None:
            y_hat = SigmoidWithBCEloss.forward(out)
            y_hat = y_hat.reshape(out.shape[0])            
            return y_hat
        else: 
            loss, cache['SigmoidWithBCEloss'] = SigmoidWithBCEloss.forward(out, y)
            return cache, loss
    
    def backward(self, cache, L2_norm=0.0):
        """
        Compute the gradients
        
        Args:
            cache: (dict) Values needed to compute gradients
            L2_norm: (int) L2 normalization coefficient
            
        Returns:
            grads: (dict) Dictionary mapping parameter names to gradients of model parameters
            
        """
        dh3 = SigmoidWithBCEloss.backward(cache['SigmoidWithBCEloss'])
        ### CODE HERE ###
                        #  y_hat이 안들어갔어.
        dout = SigmoidWithBCEloss.backward(cache['SigmoidWithBCEloss'])  
        dz2, dW3, db3 = Linear.backward(cache['out'], dout)
        dW3 = dW3 + 2 * L2_norm * self.model['W3']  
                          
        dh2 = Tanh.backward(cache['z2'], dz2)  
        dz1, dW2, db2 = Linear.backward(cache['h2'], dh2) 
        dW2 = dW2 + (2 * L2_norm * self.model['W2'])
                          
        dh1 = ReLU.backward(cache['z1'], dz1)
        dx, dW1, db1 = Linear.backward(cache['h1'], dh1)
        dW1 = dW1 + (2 * L2_norm * self.model['W1'])  
                          
        ###########################################
        grads = dict()
        grads['dout'] = dout
        grads['dW3'] = dW3
        grads['db3'] = db3
        grads['dW2'] = dW2
        grads['db2'] = db2
        grads['dW1'] = dW1
        grads['db1'] = db1

        return grads

    def train(self, X_train, y_train, X_val=None, y_val=None, learning_rate=1e-3, L2_norm=0.0, epoch=20000, print_loss=True):
        """
        Descriptions:
            Train the neural network using gradient descent.
        
        Args:
            X_train: (numpy array) training data (N, D)
            X_val: (numpy array) validation data (N, D)
            y_train: (numpy array) training labels (N,)
            y_val: (numpy array) valiation labels (N, )
            y_pred: (numpy array) Predicted target (N,)
            learning_rate: (float) Scalar giving learning rate for optimization
            L2_norm: (float) Scalar giving regularization strength.
            epoch: (int) Number of epoch to take
            print_loss: (bool) if true print loss during optimization

        Returns:
            A dictionary giving statistics about the training process
        """

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(epoch):
            ### CODE HERE ###
           # y_hat = self.forward(X_train)
            cache, loss = self.forward(X_train, y_train)
            
            loss = loss + L2_norm * (np.sum(self.model['W1']**2) + np.sum(self.model['W2']**2) + np.sum(self.model['W3']**2))
            grads = self.backward(cache, L2_norm=L2_norm)
            lr = learning_rate
            
           # y_hat = y_hat - lr * grads['dout']
            self.model['W1'] = self.model['W1'] - lr * grads['dW1']
            self.model['b1'] = self.model['b1'] - lr * grads['db1']
            self.model['W2'] = self.model['W2'] - lr * grads['dW2']
            self.model['b2'] = self.model['b2'] - lr * grads['db2']
            self.model['W3'] = self.model['W3'] - lr * grads['dW3']
            self.model['b3'] = self.model['b3'] - lr * grads['db3']
            
            ################# 
            if (it+1) % 1000 == 0:
                loss_history.append(loss)

                y_train_pred = self.predict(X_train)
                train_acc = np.average(y_train==y_train_pred)
                train_acc_history.append(train_acc)
                
                if X_val is not None:
                    y_val_pred = self.predict(X_val)
                    val_acc = np.average(y_val==y_val_pred)
                    val_acc_history.append(val_acc)
            if print_loss and (it>10) == 0:
                print(f"Loss (epoch {it+1}): {loss}")
        
            if print_loss and (it+1) % 1000 == 0:
                print(f"Loss (epoch {it+1}): {loss}")

         
        if X_val is not None:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history,
            }
        else:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
            }

    def predict(self, X):
        ### CODE HERE ###
        y_pred = self.forward(X)
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        
        return y_pred
        #################  