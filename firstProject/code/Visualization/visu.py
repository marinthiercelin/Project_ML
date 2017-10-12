import sys
import os
sys.path.insert(0, '/home/marin/Documents/Studies/EPFL_MA1/ML/Project_ML/firstProject/code/Helpers')
import proj1_helpers as helper
import matplotlib.pyplot as plt

y,x, ids = helper.load_csv_data('/home/marin/Documents/Studies/EPFL_MA1/ML/Project_ML/firstProject/data/train.csv', True)

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

path = '/home/marin/Documents/Studies/EPFL_MA1/ML/Project_ML/firstProject/data/Figures/2D/'
produce2DFigures(x,y,folder_path = path, save = True)
