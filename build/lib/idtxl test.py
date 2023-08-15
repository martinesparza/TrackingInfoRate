# Import classes
from idtxl.multivariate_te import MultivariateTE
from idtxl.bivariate_te import BivariateTE
from idtxl.data import Data
from idtxl.visualise_graph import plot_network
from idtxl.estimators_jidt import JidtKraskovCMI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csv = np.genfromtxt ('exampleData.csv', delimiter=",")
cursor = csv[0:418,0:29]
colour = csv[0:418,30:59]
target = csv[0:418,60:89]
cursor = np.delete(cursor,7,1)#trial with only nans
colour = np.delete(colour,7,1)
target = np.delete(target,7,1)
d = np.stack((cursor,colour,target))
d = np.nan_to_num(d)
data = Data(d, dim_order='psr')

estimator = JidtKraskovCMI()
test = estimator.estimate(target[1:,1],cursor[1:,1])

if (False):
    settings = {'cmi_estimator': 'JidtKraskovCMI',
                'current_value': [(0,20)],
                'selected_vars_sources': [(2,4),(2,5)]}
    network_analysis = BivariateTE()
    results = network_analysis.analyse_single_target(settings=settings,
                                                     target=0,
                                                     data=data)

    settings = {'cmi_estimator': 'JidtKraskovCMI','max_lag_sources': 20,
                'min_lag_sources': 1}
    network_analysis = MultivariateTE()
    results = network_analysis.analyse_single_target(settings=settings,
                                                     data=data,
                                                     target=0)

    #Plot inferred network to console and via matplotlib
    results.print_edge_list(weights='max_te_lag', fdr=False)
    plot_network(results=results, weights='max_te_lag', fdr=False)
    plt.show()