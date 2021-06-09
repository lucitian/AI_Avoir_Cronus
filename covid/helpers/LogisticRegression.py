import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import metrics

class LogisticRegression(object):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, debug = False):
        self.X = np.matrix(X)
        self.y = np.matrix(y).reshape(X.shape[0],1)

        self.parameters =  np.zeros((X.shape[1], 1))
        self.parameters_history = []
        
        self._cost_history = []

        self.X_test = []
        self.X_train = []
        self.y_test = []
        self.y_train = []

        self.debug = debug
    
    @property
    def cost_history(self):
        return self._cost_history
    
    def predict(self, features):
        predicted = 1 / (1 + np.exp(-(features * self.parameters)))

        return predicted
    
    def training_split(self, test_size = 1):
        test_size_percentage = int(len(self.X) * test_size)

        self.X_test = self.X[0:test_size_percentage]

        self.X_train = self.X[test_size_percentage:]

        self.y_test = self.y[0:test_size_percentage]

        self.y_train = self.y[test_size_percentage:]
    
    def cost_function(self):
        m = len(self.X)
        prediction = self.predict(self.X)

        return 1/m * ((-self.y.T * np.log(prediction) - ((1 - self.y).T) * np.log(1 - prediction)))[0,0]
    
    def cost_gradient(self):
        m = len(self.X)

        return 1/m * np.dot(self.X.T, (self.predict(self.X) - self.y))

    def train_model(self, iterations = 1000, learning_step = 0.05):
        self.parameters_history = self._cost_history = np.zeros((iterations,))

        for i in range(iterations):
            self.parameters = self.parameters - (learning_step * self.cost_gradient())
            self._cost_history[i] = self.cost_function()
            self.parameters_history[i] = self.parameters[0,0]
    
    def save_model(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    def accuracy(self, predicted):
        return metrics.accuracy_score(self.y_test, predicted)

    def visualize_cost(self, type = "show"):
        if self.debug:
            plt.plot(list(range(len(self._cost_history))), self._cost_history, color="red")
            plt.xlabel("Iteration")
            plt.ylabel("J(θ)")
            plt.title("J(θ) per Iters")
            plt.show() if type == "show" else plt.savefig("cost_per_iters") if type == "save" else True
            return 
        
        raise ValueError("Debug mode must be on to run this.")

    def visualize_params(self, type = "show"):
        if self.debug:
            plt.plot(self.parameters_history, self._cost_history, color="red")
            plt.xlabel("θ")
            plt.ylabel("J(θ)")
            plt.title("Theta vs Cost")
            plt.show() if type == "show" else plt.savefig("cost_per_params") if type == "save" else True
            return 
        
        raise ValueError("Debug mode must be on to run this.")