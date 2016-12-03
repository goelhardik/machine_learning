import numpy as np
import sys, time

class LinearRegression():
    def __init__(self, n):
        self.w = np.ones((n - 1,))
        self.alpha = 0.01
        self.tol = 0.0001

    def predict(self, x):
        return np.dot(x, self.w)

    def objective(self, x, y):
        y_hat = self.predict(x)
        return np.mean(np.square(y_hat - y))

    def fit(self, x, y):
        # keep iterating until change in objective is < tol
        oldobj = float('inf')
        newobj = self.objective(x, y)
        it = 0
        while (oldobj - newobj > self.tol):
            it += 1
            # predict y
            y_hat = self.predict(x)
            # calculate gradient
            grad = np.mean((y_hat - y)[:, None] * x, axis = 0)
            # update weights
            self.w -= self.alpha * grad
            # evaluate new obj
            oldobj = newobj
            newobj = self.objective(x, y)
            print("Iteration {} OBJ {}".format(it, newobj))

def train_test_split(data, target):
    # perform a 10% split
    n = data.shape[0]
    np.random.shuffle(data)
    ind = int(0.9 * n)
    x_train, y_train = data[:ind, :target], data[:ind, target]
    x_test, y_test = data[ind:, :target], data[ind:, target]
    return x_train, y_train, x_test, y_test

def z_score(data, target):
    y = data[:, target]
    means = np.mean(data, axis = 0)
    stds = np.std(data, axis = 0)
    data = (data - means) / stds
    data[:, target] = y
    return data

if __name__ == '__main__':
    datafile = sys.argv[1]
    data = np.loadtxt(datafile, delimiter = ',', comments = None)
    # normalize data
    data = z_score(data, -1)
    # insert column of 1s in x
    data = np.insert(data, 0, np.ones(data.shape[0]), axis = 1)
    # split data
    x_train, y_train, x_test, y_test = train_test_split(data, -1)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    start = time.time()
    model = LinearRegression(data.shape[1])
    model.fit(x_train, y_train)
    y_hat_test = model.predict(x_test)
    print("Test error by Gradient Descent {} Time taken "
          "{}".format(np.mean(np.square(y_hat_test - y_test)), time.time() - start))

    start = time.time()
    # get w by normal equations
    w = np.dot(np.linalg.pinv(np.dot(x_train.T, x_train)), np.dot(x_train.T, y_train))
    model.w = w
    y_hat_test = model.predict(x_test)
    print("Test error by normal equations {} Time taken "
          "{}".format(np.mean(np.square(y_hat_test - y_test)), time.time() - start))
