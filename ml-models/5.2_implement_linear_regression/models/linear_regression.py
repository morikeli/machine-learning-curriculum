import numpy as np

class LinearRegression:
    """ This is an custom object to mimic and implement linear regression model in Machine Learning. """

    def __init__(self, learning_rate, iterations):  # hyperparameters: "learning_rate" & "iterations"
       self.learning_rate = learning_rate
       self.iterations = iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape    # no. of rows & columns

        # initiate weight & bias
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X  # years of experience in the salary dataset, i.e. DIR_PATH="data/salary_data.csv"
        self.Y = Y  # salary in the salary dataset

        # implementing gradient descent
        for i in range(self.iterations):
            self.update_weights()


    def update_weights(self):
        Y_prediction = self.predict(self.X)

        # calculate gradients
        dw = -(2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m

        db = -2 * np.sum(self.Y - Y_prediction) / self.m

        # update weights
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db


    def predict(self, X):
        return X.dot(self.w) + self.b

