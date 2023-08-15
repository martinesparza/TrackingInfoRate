# Import classes
#import matplotlib.pyplot as plt
import information_transfer as it
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", type=str)
args = parser.parse_args()
#args.filename = 'exampleData.csv'

# functions for dealing with nans
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_nans(y):
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y

# load data
def load_data(filename):
    csv = np.genfromtxt (filename, delimiter=",")
    cursor = csv[1:418,0:30]
    colour = csv[1:418,30:60]
    target = csv[1:418,60:90]

    # remove nans
    allTrials = list(range(30))
    nanTrials = np.where(np.all(np.isnan(cursor), axis=0))#trials with only nans
    for tr in nanTrials[0]:
        cursor = np.delete(cursor,tr,1)
        colour = np.delete(colour,tr,1)
        target = np.delete(target,tr,1)
        allTrials.pop(tr)
    for idx, tr in enumerate(cursor.T):
        tr = interpolate_nans(tr)
        cursor[:,idx] = tr
    return cursor, colour, target

# compute FB, FF and total info
def compute_info_transfer():
    cursor, colour, target = load_data(args.filename)
    # only target
    targetFB = []
    targetINFO = []
    order = [4, 3]
    for col in range(cursor.shape[1]):
        targetFB.append(it.compute_FB([target[:,col]],cursor[:,col],order,VMD))
        targetINFO.append(it.compute_total_info([target[:, col]], cursor[:, col], order, VMD))
    targetFF = np.array(targetINFO)-np.array(targetFB)
        #only colour
    colourFB = []
    colourINFO = []
    order = [0, 3]
    for col in range(colour.shape[1]):
        colourFB.append(it.compute_FB([colour[:,col]],cursor[:,col],order,VMD))
        colourINFO.append(it.compute_total_info([colour[:, col]], cursor[:, col], order, VMD))
    colourFF = np.array(colourINFO)-np.array(colourFB)
        #both cursor and colour
    bothFB = []
    bothINFO = []
    order = [4, 0, 3]
    for col in range(cursor.shape[1]):
        bothFB.append(it.compute_FB([target[:,col],colour[:,col]],cursor[:,col],order,VMD))
        bothINFO.append(it.compute_total_info([target[:,col],colour[:, col]], cursor[:, col], order, VMD))
    bothFF = np.array(bothINFO)-np.array(bothFB)

    zipped = list(zip(targetFB, targetFF, targetINFO, colourFB, colourFF, colourINFO, bothFB, bothFF, bothINFO))
    df = pd.DataFrame(zipped, columns=['TargetFeedback', 'TargetFeedforward', 'TargetTotalInfo',
                                       'ColourFeedback', 'ColourFeedforward', 'ColourTotalInfo',
                                       'BothFeedback', 'BothFeedforward', 'BothTotalInfo'])
    df.to_csv('output_'+args.filename,index=False)

VMD=17
compute_info_transfer()
# plotting
# t = np.array(allTrials)+6 # adding 6 to trial number because of the 6 training trials
# plt.plot(t,cursorFB,'b',t,colourFB,'g',t,bothFB,'r')
# plt.title("Feedback info")
# plt.xlabel("Trial number")
# plt.ylabel("FB")
# plt.show()
#
# plt.plot(t,cursorFF,'b',t,colourFF,'g',t,bothFF,'r')
# plt.title("Feedback info")
# plt.xlabel("Trial number")
# plt.ylabel("FB")
# plt.show()