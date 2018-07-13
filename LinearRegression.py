import numpy
import math
import matplotlib.pyplot as plt

class LinearRegression():

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.Ns = self.X.shape[0]

    def fit(self):
        self.sample = numpy.c_[numpy.ones(self.Ns), self.X]
        self.RegressionCoeff = reduce(numpy.dot,[numpy.linalg.inv(numpy.dot(numpy.transpose(self.sample), self.sample)),
                                                 numpy.transpose(self.sample), self.y])
        self.predictor = numpy.dot(self.sample, self.RegressionCoeff)

    def predict(self, X):
        self.X_int = X
        self.X_pred = numpy.c_[numpy.ones(X.shape[0]), X]
        self.y_pred = numpy.dot(self.X_pred, self.RegressionCoeff)

    def RSquare(self): #Coefficient of determination
        self.mean_y = numpy.mean(self.y)
        SStot = numpy.sum((self.y - self.mean_y) ** 2)
        SSres = numpy.sum((self.y - self.predictor) ** 2)
        SSreg = numpy.sum((self.predictor - self.mean_y) ** 2)
        self.Rsquare = 1 - (numpy.float(SSres) / numpy.float(SStot))
        return self.Rsquare
    
    def plot(self, Xlabel="X", Ylabel="Y"):
        
        self.Rsquare = self.RSquare()
        plt.figure()
        plt.scatter(self.X[:], self.y)
        plt.plot(self.X_int[:], self.y_pred)
        plt.xlabel(Xlabel, fontsize=16)
        plt.ylabel(Ylabel, fontsize=16)
        plt.title("$R^2 = %f $" % (self.Rsquare), fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_acc(self):

        plt.figure()
        plt.scatter(self.y, self.y_pred)
        plt.plot(self.y, self.y)
        plt.xlabel("Actual value", fontsize=16)
        plt.ylabel("Predicted value", fontsize=16)
        plt.tight_layout()
        plt.show()


x = numpy.array(numpy.arange(1, 11, 1))
y = x**2

Regression = LinearRegression(x, y)
Regression.fit()
Regression.predict(x)
Regression.plot("X", "y")
Regression.plot_acc()
        
        
