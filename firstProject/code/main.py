import Helpers.proj1_helpers as helper
import Visualization.visu as visu
import cleaner

y,x, ids = helper.load_csv_data('../data/train.csv', True)

path = '../data/Figures/2D/'
xCl = cleaner.cleanFeatures(x)
visu.produce2DFigures(xCl,y,folder_path = path, save = True)
