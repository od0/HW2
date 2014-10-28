import numpy as np
from scipy import stats
import csv as csv
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from math import sqrt

import config

def load_data(infile):
    arr = []
    input = csv.reader(open(infile,'rb'))
    header = input.next()
    for row in input:
        try:
            data = np.array(row)
            ttime = float(data[8])
            plong = float(data[10])
            plat = float(data[11])
        except:
            continue
        if not (config.minlong < plong < config.maxlong):
            continue
        if not (config.minlat < plat < config.maxlat):
            continue
        arr.append([ttime,plong,plat])
    return np.array(arr)

def distance(point1,point2,length):
    distance = 0
    for x in range(length):
        distance += pow((point1[x] - point2[x]), 2)
    return math.sqrt(distance)

def knn(trndata,tstdata,k):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(trndata[:,1:])
    distances,indices = nbrs.kneighbors(tstdata[:,1:])
    print "Root Mean Squared Error: ", round(sqrt(mean_squared_error(tstdata[indices,0], trndata[indices,0])),1)
    coef,pval = stats.pearsonr(tstdata[indices,0], trndata[indices,0])
    print "Correlation Coefficient: ", round(coef[0],4)
    print "Means Absolute Error: ", round(np.mean(np.sum(np.absolute(tstdata[indices,0] - trndata[indices,0]))),1)

def main():
    trndata = load_data(config.TRAIN_DATA)
    tstdata = load_data(config.TEST_DATA)
    knn(trndata,tstdata,1)

if __name__ == '__main__':
    main()