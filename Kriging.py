import numpy
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

class Kriging():

    def __init__(self, alpha=1e-2, optimizer="None", corr_function="exponential"):
        self.alpha = alpha
        self.optimizer = optimizer
        self.corr_function = corr_function.lower()

    def fit(self, X, y, theta):
        """Fit kriging model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = (n_samples, [n_output_dims])
            Target values

        Returns
        -------
        self : returns an instance of self.
        """

        self.X = X
        self.y = y
        self.theta = theta
        
        self.Ns = self.X.shape[0]
        self.Rs = numpy.zeros((self.Ns,self.Ns))
        
                
        if self.corr_function == "exponential":
            for i in xrange(self.Ns):
                for j in xrange(self.Ns):
                    d = abs(self.X[i] - self.X[j])
                    R_temp = numpy.exp(-1 * self.theta * (d ** 2))
                    self.Rs[i][j] = numpy.prod(R_temp)
                    
        self.F = numpy.ones((self.Ns,1))


        beta_1 = reduce(numpy.dot,[numpy.transpose(self.F), inv(self.Rs), self.F])
        beta_2 = reduce(numpy.dot,[numpy.transpose(self.F), inv(self.Rs), numpy.transpose(self.y)])
        self.Beta = numpy.dot(inv(beta_1), beta_2)

        

    def predict(self, X):

        self.X_pred = X
        Nn = self.X_pred.shape[0]
        self.y_pred = numpy.zeros((Nn, 1))
        self.r = numpy.zeros((Nn, self.Ns))
        
        if self.corr_function == "exponential":
            for i in xrange(Nn):
                for j in xrange(self.Ns):
                    d = abs(self.X_pred[i] - self.X[j])
                    R_temp = numpy.exp(-1 * self.theta * (d ** 2))
                    self.r[i][j] = numpy.prod(R_temp)

        
        gamma = numpy.transpose(self.y) - (self.F * self.Beta)
        
        for i in xrange(Nn):
            predictor = self.Beta + reduce(numpy.dot,[self.r[i,:], inv(self.Rs), gamma[0]])
            
            self.y_pred[i] = predictor.item(0)
        
        return self.y_pred

    def plot(self):

        plt.figure()
        plt.scatter(self.y, self.y_pred)
        plt.xlabel("Actual value", fontsize=16)
        plt.ylabel("Predicted value", fontsize=16)
        plt.tight_layout()
        plt.show()
        
   # def optimization(self):
        #if self.optimizer == "None": # Use an GEK optimization as default
            

def problem(x):
    y = (x[:,0] **2) + 25 * (numpy.sin(x[:,1]) ** 2)
    return y

sample = numpy.ones((10,2))
sample[:,0] = numpy.arange(1,11,1)
sample[:,1] = numpy.arange(1,11,1)

target = problem(sample)

x_check = numpy.ones((10,2))
x_check[:,0] = numpy.arange(1,11,1)
x_check[:,1] = numpy.arange(1,11,1)

model = Kriging()
model.fit(sample, target, 0.1)
model.predict(x_check)
model.plot()

plt.figure()
plt.scatter(sample[:,1], target)
plt.plot(x_check[:,1], model.y_pred)

#plt.figure()
#plt.scatter(target, model.y_pred)
plt.show()

