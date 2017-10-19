import os
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/marin/Documents/Studies/EPFL_MA1/ML/Project_ML/firstProject/code/Helpers')
import Helpers.implementations as tools
import numpy as np

def make2DScatter(x,i,j, colors):
    plt.scatter(x[:,i],x[:,j], c=colors)
    plt.title("i = " + str(i) + " j = " + str(j))
    plt.xlabel("x("+str(i)+")")
    plt.ylabel("x("+str(j)+")")

def produce2DFigures(x,y,folder_path = '', save = False):
    colors = [ ('b' if(yel == 1) else 'r') for yel in y]
    for i in range(x.shape[1]):
        if(save):
            path = folder_path + 'index'+str(i)+'/'
            if not os.path.exists(path):
                os.makedirs(path)
        for j in range(x.shape[1]):
            if(not j == i):
                make2DScatter(x,i,j,colors)
                if(save):
                    plt.savefig(path + '2D_'+str(i)+'_'+str(j)+'.png')
                else:
                    plt.show()
                plt.gcf().clear()

def produce1DFigures(x, y, folder_path = '', save = False):
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
    plt.scatter(x[:,i],y, alpha = 0.05)
    plt.title("i = " + str(i) )
    plt.xlabel("x("+str(i)+")")
    plt.ylabel("y")

def produce1DFiguresWithLinearRegression(x,y, folder_path, degrees):
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
            make1DScatter(x,y,i)
            plt.savefig(folder_path + 'i='+str(i)+'d='+str(deg)+'.png')
            plt.gcf().clear()
            print("w ", i ,deg, w)
