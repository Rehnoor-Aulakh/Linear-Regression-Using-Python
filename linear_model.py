import numpy as np

class LinearModel:
    def __init__(self, num_features):
        self.num_features= num_features
        self.W = np.random.randn(num_features,1)
        self.b = np.random.randn()
        
    def forward_pass(self, X):
        y_hat = self.b + np.dot(X,self.W)
        return y_hat
    
    def compute_loss(self, y_hat, y_true):
        return np.sum(np.square(y_hat-y_true))/(2*y_hat.shape[0])
    
    def backward_pass(self, X, y_true, y_hat):
        m = y_true.shape[0]
        db = (1/m)*np.sum(y_hat - y_true)
        dW = (1/m)*np.sum(np.dot(np.transpose(y_hat-y_true), X), axis=0)
        return dW,db
    
    def update_params(self, dW, db, lr):
        self.W = self.W - lr*np.reshape(dW, (self.num_features, 1))
        self.b = self.b - lr * db
        
    def train(self, x_train, y_train, iterations, lr):
        losses = []
        for i  in range(0, iterations):
            y_hat= self.forward_pass(x_train)
            loss = self.compute_loss(y_hat, y_train)
            dW , db = self.backward_pass(x_train, y_train, y_hat)
            self.update_params(dW,db,lr)
            losses.append(loss)
            if i%int(iterations/10)==0:
                print('Iter: {}, Loss: {:.4f}'.format(i,loss))
                
        return losses