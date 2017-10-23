import os
import matplotlib.pyplot as plt
import sys
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
            data = tools.build_poly(x[:,i], deg)
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
