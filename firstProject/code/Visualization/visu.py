import os
import matplotlib.pyplot as plt
#import sys
#sys.path.insert(0, '/home/marin/Documents/Studies/EPFL_MA1/ML/Project_ML/firstProject/code/Helpers')
#import implementations as tools
import numpy as np

def make2DScatter(x_i,x_j,i,j, colors):
    """Create a scatter plot with x_i and x_j with the given colors."""
    plt.scatter(x_i,x_j, c=colors)
    plt.title("i = " + str(i) + " j = " + str(j))
    plt.xlabel("x("+str(i)+")")
    plt.ylabel("x("+str(j)+")")

def produce2DFigures(x,y,folder_path):
    """Produce all 2D scatter plots for x and store them in folder_path.

    Select a color for each sample depending of his category (y).
    Create a directory at the given path if it doesn't exist yet.
    Run through x and create the scatter plot for each pair of features with make2DScatter.
    Save those plots in the directory.
    """
    colors = [ ('b' if(yel == 1) else 'r') for yel in y]
    for i in range(x.shape[1]):
        path = folder_path + 'index'+str(i)+'/'
        if not os.path.exists(path):
            os.makedirs(path)
        for j in range(x.shape[1]):
            if(not j == i):
                make2DScatter(x[:,i],x[:,j],i,j,colors)
                plt.savefig(path + '2D_'+str(i)+'_'+str(j)+'.png')
                plt.gcf().clear()

def produce2DFiguresWOMissing(x,y,folder_path):
    """Produce all 2D scatter plots for x after ignoring all missing values (-99) and store them in folder_path.

    Select a color for each sample depending of his category (y).
    Create a directory at the given path if it doesn't exist yet.
    Select all indices where x != -999.
    For each pair of features, draw the scatter plot for all samples that have a value for both features with make2DScatter.
    Save those plots in the directory.
    """
    colors = [ ('b' if(yel == 1) else 'r') for yel in y]
    for i in range(x.shape[1]):
        path = folder_path + 'index'+str(i)+'/'
        if not os.path.exists(path):
            os.makedirs(path)
        x_i = x[:,i]
        ind_1 = np.where(x_i != -999)
        for j in range(x.shape[1]):
            if(not j == i):
                x_j = x[:, j]
                ind_2 = np.where(x_j != -999)
                rest = np.intersect1d(ind_1,ind_2)
                x_i_j = x_i[rest]
                x_j = x_j[rest]
                col = np.array(colors)
                col = col[rest]
                make2DScatter(x_i_j,x_j,i,j,col)
                plt.savefig(path + '2D_'+str(i)+'_'+str(j)+'.png')
                plt.gcf().clear()

def make1DScatter(x,y,i):
    """Create a scatter plot with x and y with the given colors."""
    plt.scatter(x,y, alpha = 0.05)
    plt.title("i = " + str(i) )
    plt.xlabel("x("+str(i)+")")
    plt.ylabel("y")

def produce1DFigures(x, y, folder_path = '', save = False):
    """Produce all 1D scatter plots for x and store them in folder_path.

    Create a directory at the given path if it doesn't exist yet.
    Run through x and create the scatter plot for each feature depending on the category of the sample.
    Save those plots in the directory.
    """
    if(save):
        folder_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    for i in range(x.shape[1]):
        make1DScatter(x,y,i)
        if(save):
            plt.savefig(folder_path + '1D_'+str(i)+'.png')
        else:
            plt.show()
        plt.gcf().clear()

def make1DScatter(x,y,i):
    plt.scatter(x,y, alpha=0.1)
    plt.title("i = " + str(i) )
    plt.xlabel("x("+str(i)+")")
    plt.ylabel("y")

def produce1DFiguresWOMissing(x,y, folder_path):
    """Produce all 1D scatter plots for x and store them in folder_path.

    Create a directory at the given path if it doesn't exist yet.
    Run through x and create the scatter plot for each feature for samples that have a value in this feature depending on the category of the sample.
    Save those plots in the directory.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i in range(x.shape[1]):
        data = x[:,i]
        ind = np.where(data != -999)
        data = data[ind]
        y_d = y[ind]
        make1DScatter(data,y_d,i)
        plt.savefig(folder_path + 'i='+str(i)+'.png')
        plt.gcf().clear()


def produce1DFiguresWithLinearRegression(x,y, folder_path, degrees):
    """Produce the same plots than produce1DFigures but add a drawing of the linear regression.

    Create a directory at the given path if it doesn't exist yet.
    Compute the linear regression
    Draw the 1D scatter plot as in produce1DFigures on top of the regression line.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    max_iters = 100
    gamma = 0.8
    for deg in degrees :
        initial_w = np.array((deg+1)*[0])
        for i in range(x.shape[1]):
            data = tools.build_poly(data, deg)
            loss, w = tools.least_squares_GD(y,data, initial_w, max_iters, gamma)
            lin = np.linspace(max(-1000,np.min(x[:,i])), min(1000,np.max(x[:,i])), 1000)
            tmp = np.array([np.power(lin, d) for d in range(deg +1)])
            tmp = tmp.T
            res = np.dot(tmp,w)
            plt.plot(lin,res)
            make1DScatter(x[:,i],y,i)
            plt.savefig(folder_path + 'i='+str(i)+'d='+str(deg)+'.png')
            plt.gcf().clear()
            print("w ", i ,deg, w)

def produce1DFiguresWithLinearRegressionWOMissing(x,y, folder_path, degrees):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    max_iters = 100
    gamma = 0.8
    for deg in degrees :
        initial_w = np.array((deg+1)*[0])
        for i in range(x.shape[1]):
            x_i = x[:,i]
            ind = np.where(x_i != -999)
            x_i = x_i[ind]
            y_d = y[ind]
            data = tools.build_poly(x_i, deg)
            loss, w = tools.least_squares_GD(y_d,data, initial_w, max_iters, gamma)
            lin = np.linspace(max(-1000,np.min(x[:,i])), min(1000,np.max(x[:,i])), 1000)
            tmp = np.array([np.power(lin, d) for d in range(deg +1)])
            tmp = tmp.T
            res = np.dot(tmp,w)
            plt.plot(lin,res)
            make1DScatter(x_i,y_d,i)
            axes = plt.gca()
            axes.set_ylim([-1,1])
            plt.savefig(folder_path + 'i='+str(i)+'d='+str(deg)+'.png')
            plt.gcf().clear()
            print("w ", i ,deg, w)


def produce2DWithBoundary(y,x,folder_path):
    colors = [ ('b' if(yel == 1) else 'r') for yel in y]
    #parameters
    gamma = 0.1
    initial_w = [0,0]
    for i in range(2,x.shape[1]):
        path = folder_path + 'index'+str(i)+'/'
        if not os.path.exists(path):
            os.makedirs(path)
        x_i = x[:,i]
        ind_1 = np.where(x_i != -999)
        for j in range(x.shape[1]):
            if(not j == i):
                x_j = x[:, j]
                ind_2 = np.where(x_j != -999)
                rest = np.intersect1d(ind_1,ind_2)
                x_i_j = x_i[rest]
                x_j = x_j[rest]
                tx = np.c_[x_i_j,x_j]
                logistic_regression_gradient_descent_demo(y[rest],tx,i,j,path)
                plt.gcf().clear()

def logistic_regression_gradient_descent_demo(y, x,i,j,folder_path):
    # init parameters
    max_iter = 1000
    gamma = 0.02
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.array(tx.shape[1]*[0])

    # start the logistic regression
    loss, w = tools.logistic_regression(y,tx,w, max_iter, gamma)
    # visualization
    show2DBoundary(y, x, i, j, w, folder_path + "i=" + str(i) + "j=" + str(j))

def show2DBoundary(y, x,i,j, w, save_name):
    """visualize the raw data as well as the classification result."""
    fig = plt.figure()
    # plot raw data
    ax1 = fig.add_subplot(1, 2, 1)
    males = np.where(y == 1)
    females = np.where(y == 0)
    ax1.scatter(
        x[males, 0], x[males, 1],
        marker='.', color=[0.06, 0.06, 1], s=20)
    ax1.scatter(
        x[females, 0], x[females, 1],
        marker='*', color=[1, 0.06, 0.06], s=20)
    ax1.set_xlabel("x["+str(i)+"]")
    ax1.set_ylabel("x["+str(j)+"]")
    ax1.grid()
    # plot raw data with decision boundary
    ax2 = fig.add_subplot(1, 2, 2)
    height = np.arange(
        np.min(x[:, 0]), np.max(x[:, 0]) +0.01, step=0.01)
    weight = np.arange(
        np.min(x[:, 1]), np.max(x[:, 1]) + 0.01, step=0.01)
    hx, hy = np.meshgrid(height, weight)
    hxy = (np.c_[hx.reshape(-1), hy.reshape(-1)])
    x_temp = np.c_[np.ones((hxy.shape[0], 1)), hxy]
    prediction = x_temp.dot(w) > 0.5
    prediction = prediction.reshape((weight.shape[0], height.shape[0]))
    ax2.contourf(hx, hy, prediction, 1)
    ax2.scatter(
        x[males, 0], x[males, 1],
        marker='.', color=[0.06, 0.06, 1], s=20)
    ax2.scatter(
        x[females, 0], x[females, 1],
        marker='*', color=[1, 0.06, 0.06], s=20)
    ax2.set_xlabel("x["+str(i)+"]")
    ax2.set_ylabel("x["+str(j)+"]")
    ax2.set_xlim([min(x[:, 0]), max(x[:, 0])])
    ax2.set_ylim([min(x[:, 1]), max(x[:, 1])])
    plt.tight_layout()
    plt.savefig(save_name)
