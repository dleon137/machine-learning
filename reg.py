import numpy as np

# sigmoid function
def g(z):
    return 1. / (1. + np.exp(-z))

# cost function
def J(theta, X, y, lambda_reg=0., log=True):
    m = y.size
    if log:
        hx = g(np.dot(X, theta))
        val = (-np.dot(y, np.log(hx)) - np.dot(1 - y, np.log(1 - hx))) / m
    else:
        hx = np.dot(X, theta)
        val = 0.5 * np.mean((hx - y)**2)
    reg = 0.5 * lambda_reg * np.mean(theta[1:]**2)
    return val + reg

# linear cost function
def Jlin(theta, X, y, lambda_reg=0.):
    return J(theta, X, y, lambda_reg, False)

# logistic cost function
def Jlog(theta, X, y, lambda_reg=0.):
    return J(theta, X, y, lambda_reg, True)

# gradient of cost function
def grad(theta, X, y, lambda_reg=0., log=True):
    m = y.size
    hx = np.dot(X, theta)
    if log:
        hx = g(hx)
    val = np.dot(hx - y, X) / m
    val[1:] += lambda_reg / m * theta[1:]
    return val

# gradient of linear cost function
def gradlin(theta, X, y, lambda_reg=0.):
    return grad(theta, X, y, lambda_reg, False)

# gradient of logistic cost function
def gradlog(theta, X, y, lambda_reg=0.):
    return grad(theta, X, y, lambda_reg, True)

