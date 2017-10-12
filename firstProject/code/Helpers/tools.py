import numpy as np

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = y.shape[0]
    e = y - np.dot(tx,w)
    return (-1.0*(np.dot(tx.T,e)))/N


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        w = w - gamma*gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y,tx,30):
            gradient = compute_gradient(batch_y,batch_tx,w)
            loss = compute_loss(batch_y,batch_tx,w)
            w = w - gamma*gradient
            # store w and loss
            ws.append(w)
            losses.append(loss)
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws


def compute_mse(y, tx, w):
    """compute the loss by mse."""
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
    

def least_squares(y, tx):
    """calculate the least squares solution."""
    xt = tx.T
    gram = np.dot(xt,tx)
    ft = np.dot(xt,y)
    w = np.linalg.lstsq(gram, ft)[0]
    loss = compute_loss(y,tx,w)
    return loss, w

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    return np.array([[np.power(xi,n) for n in range(degree + 1)] for xi in x])

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    ind = list(range(y.shape[0]))
    np.random.shuffle(ind)
    x_shuffled = x[ind]
    y_shuffled = y[ind]
    limit = int(np.floor(y.shape[0]*ratio))
    return x_shuffled[:limit],y_shuffled[:limit],x_shuffled[limit:],y_shuffled[limit:]

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    xt = tx.T
    gram = np.dot(xt,tx)
    gram += (2.0*y.shape[0]*lambda_)*np.identity(gram.shape[0])
    ft = np.dot(xt,y)
    return np.linalg.lstsq(gram, ft)[0]
