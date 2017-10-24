import numpy as np

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = y.shape[0]
    e = y - np.dot(tx,w)
    return (-1.0*(np.dot(tx.T,e)))/N

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y,tx,w)
        loss = compute_mse(y,tx,w)
        w = w - gamma*gradient
    loss = compute_mse(y,tx,w)
    return loss, w

def batch_iter(y, tx, batch_size, seed, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset.

    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        np.random.seed(seed)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma, seed=1):
    """Stochastic gradient descent algorithm using mse."""
    w = initial_w
    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y,tx,batch_size, seed):
            gradient = compute_gradient(batch_y,batch_tx,w)
            loss = compute_mse(batch_y,batch_tx,w)
            w = w - gamma*gradient
    loss = compute_mse(y, tx, w)
    return loss, w

#loss functions
def compute_mse(y, tx, w):
    """Compute the loss with the mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_mae(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - np.dot(tx,w)
    e = np.absolute(e)
    N = y.shape[0]
    return e.sum()/(N*1.0)

#least squares with mse
def least_squares(y, tx):
    """Calculate the least squares solution."""
    xt = tx.T
    gram = np.dot(xt,tx)
    ft = np.dot(xt,y)
    w = np.linalg.lstsq(gram, ft)[0]
    loss = compute_mse(y,tx,w)
    return loss, w

def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    return np.array([[np.power(xi,n) for n in range(degree + 1)] for xi in x])

#used to have train and test set
def split_data(x, y, ratio, seed=1):
    """Split the data x and y with a given ratio.

    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    np.random.seed(seed)
    ind = list(range(y.shape[0]))
    np.random.shuffle(ind)
    x_shuffled = x[ind]
    y_shuffled = y[ind]
    limit = int(np.floor(y.shape[0]*ratio))
    return x_shuffled[:limit],y_shuffled[:limit],x_shuffled[limit:],y_shuffled[limit:]

#ridge regression
def ridge_regression(y, tx, lambda_):
    """Implement ridge regression."""
    xt = tx.T
    gram = np.dot(xt,tx)
    gram += (2.0*y.shape[0]*lambda_)*np.identity(gram.shape[0])
    ft = np.dot(xt,y)
    w = np.linalg.lstsq(gram, ft)[0]
    loss = compute_mse(y, tx, w)
    return loss, w

#for cross validation
def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

#for logistic regression
def sigmoid(t):
    """Apply sigmoid function on t."""
    exp_inv = np.exp(-1.0*t)
    return np.divide(1,1+exp_inv)

# regular regression
def calculate_loss(y, tx, w):
    """Compute the cost by negative log likelihood."""
    fx_list = np.dot(tx,w)
    fx_trans_1 = np.logaddexp(0,fx_list)
    fx_trans_2 = np.multiply(y,fx_list)
    res = fx_trans_1 - fx_trans_2
    return res.sum()

def calculate_gradient(y, tx, w):
    """Compute the gradient of loss."""
    fx_sigma = sigmoid(np.dot(tx,w))
    mul = fx_sigma - y
    return np.dot(tx.T,mul)

def logistic_regression_step(y, tx, w, gamma):
    """Execute on step of the logistic regression.

    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    grad = calculate_gradient(y,tx,w)
    print("grad",grad)
    w = w - gamma*grad
    return w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        w = logistic_regression_step(y,tx,w,gamma)
        print("w",w)
    loss = calculate_loss(y,tx,w)
    return loss, w

def logistic_regression_sgd(y, tx, initial_w, batch_size, max_iters, gamma, seed = 1):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y,tx,batch_size, seed):
            w = logistic_regression_step(batch_y,batch_tx,w,gamma)
    loss = calculate_loss(y, tx, w)
    return loss, w

#regularized logistic regression
def reg_calculate_loss(y, tx, lambda_, w):
    """Compute the cost by negative log likelihood."""
    return calculate_loss(y,tx,w) + (lambda_/2.0)*np.dot(w.T,w)

def reg_calculate_gradient(y, tx, lambda_, w):
    """Compute the gradient of loss."""
    return calculate_gradient(y,tx,w) + (lambda_)*w


def reg_logistic_regression_step(y, tx, lambda_, w, gamma):
    """Execute one step of the regularized logistic regression.

    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    grad = reg_calculate_gradient(y,tx, lambda_, w)
    w = w - gamma*grad
    return w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        w = reg_logistic_regression_step(y,tx,lambda_,w,gamma)
        """if(n_iter % (max_iters/10) == 0):
            loss = reg_calculate_loss(y, tx, lambda_, w)
            print(n_iter," loss ",loss)
            print(n_iter," w ", w)"""

    loss = reg_calculate_loss(y, tx, lambda_, w)
    return loss, w

def reg_logistic_regression_sgd(y, tx, lambda_, initial_w, batch_size, max_iters, gamma, seed = 1):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y,tx,batch_size, seed):
            w = reg_logistic_regression_step(batch_y,batch_tx,lambda_,w,gamma)
    loss = reg_calculate_loss(y, tx, lambda_, w)
    return loss, w
