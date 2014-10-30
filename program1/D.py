import numpy as np
from scipy import stats
import csv as csv
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
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
            tdist = float(data[9])
            plong = float(data[10])
            plat = float(data[11])
            ptime = hms_to_hours(data[5][-8:])
        except:
            continue
        if not (config.minlong < plong < config.maxlong):
            continue
        if not (config.minlat < plat < config.maxlat):
            continue
        arr.append([ttime,plong,plat,ptime,tdist])
    return np.array(arr)

def hms_to_hours(t):
    h, m, s = [float(i) for i in t.split(':')]
    return h + m/60 + s/3600

def distance(point1,point2,length):
    distance = 0
    for x in range(length):
        distance += pow((point1[x] - point2[x]), 2)
    return math.sqrt(distance)

def scale_data(trndata,tstdata):
    trndata[:,1:] *= preprocessing.normalize(trndata[:,1:])
    tstdata[:,1:] *= preprocessing.normalize(tstdata[:,1:])
    trndata[:,1:] *= preprocessing.scale(trndata[:,1:])
    tstdata[:,1:] *= preprocessing.scale(tstdata[:,1:])
    return trndata,tstdata

def knn(trndata,tstdata,k): 
    indices = []
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(trndata[:20000,1:])
    for x in range(tstdata.shape[0]):
        if x > 10000:
            break
        new_distance,new_index = nbrs.kneighbors(tstdata[x,1:])
        indices.append(new_index[0])

    return indices

def measure(trndata,tstdata,indices):
    print "Root Mean Squared Error: ", round(sqrt(mean_squared_error(tstdata[indices,0], trndata[indices,0])),1)
    coef,pval = stats.pearsonr(tstdata[indices,0], trndata[indices,0])
    print "Correlation Coefficient: ", round(coef[0],4)
    print "Mean Absolute Error: ", round(np.mean(np.absolute(tstdata[indices,0] - trndata[indices,0])),1)

def main():
    trndata = load_data(config.TRAIN_DATA)
    tstdata = load_data(config.TEST_DATA)
    #print trndata[:10,1:]
    print "Results using original data:"
    indices = knn(trndata,tstdata,1)
    measure(trndata,tstdata,indices)
    trndata_scaled, tstdata_scaled = scale_data(trndata,tstdata)
    #print trndata[:10,1:]
    print "Results using scaled data:"
    indices = knn(trndata_scaled,tstdata_scaled,1)
    measure(trndata,tstdata,indices)
    #print trndata.shape[1]

if __name__ == '__main__':
    main()