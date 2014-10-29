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
	for x in range(trndata.shape[1]):
		if x:
			trndata[:,x] *= np.mean(trndata[:,x])
			tstdata[:,x] *= np.mean(tstdata[:,x])
	return trndata,tstdata

def knn(trndata,tstdata,k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(trndata[:2000,1:])
    distances,indices = nbrs.kneighbors(tstdata[:1000,1:])
    print "Root Mean Squared Error: ", round(sqrt(mean_squared_error(tstdata[indices,0], trndata[indices,0])),1)
    coef,pval = stats.pearsonr(tstdata[indices,0], trndata[indices,0])
    print "Correlation Coefficient: ", round(coef[0],4)
    print "Mean Absolute Error: ", round(np.mean(np.absolute(tstdata[indices,0] - trndata[indices,0])),1)

def main():
    trndata = load_data(config.TRAIN_DATA)
    tstdata = load_data(config.TEST_DATA)
    #print trndata[:10,1:]
    print "Results using original data:"
    knn(trndata,tstdata,1)
    trndata_scaled, tstdata_scaled = scale_data(trndata,tstdata)
    #print trndata[:10,1:]
    print "Results using scaled data:"
    knn(trndata_scaled,tstdata_scaled,1)
    #print trndata.shape[1]

if __name__ == '__main__':
    main()