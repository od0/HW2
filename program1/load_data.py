import csv as csv
import numpy as np
from collections import defaultdict
import config

def A_old(infile):
    #from pprint import pprint
    arr1 = []
    arr3 = []
    data = []
    input = csv.reader(open(infile,'rb'))
    header = input.next()
    testbreak = 0
    for row in input:
        testbreak += 1
        if testbreak > 10000:
            break
        data = np.array(row)
        passcount = float(data[7])
        plong = float(data[10])
        plat = float(data[11])
        if not (-75 < plong < -70):
            continue
        if not (40 < plat < 41):
            continue
        if passcount == 1:
            arr1.append([plong,plat])
        if passcount == 3:
            arr3.append([plong,plat])
    arr1 = np.array(arr1)
    arr3 = np.array(arr3)
    return arr1, arr3

def transform(plong,plat,minlong,maxlong,minlat,maxlat):
    return round((plong-minlong)*config.bins/(maxlong-minlong),0),round((plat-minlat)*config.bins/(maxlat-minlat),0)

def A(infile):
    count_trips = defaultdict(int)
    count_1 = defaultdict(int)
    count_3  = defaultdict(int)
    minlong = -75
    maxlong = -70
    minlat = 40
    maxlat = 41
    input = csv.reader(open(infile,'rb'))
    header = input.next()
    testbreak = 0
    for row in input:
        #testbreak += 1
        #if testbreak > 10000:
        #    break
        data = np.array(row)
        passcount = float(data[7])
        plong = float(data[10])
        plat = float(data[11])
        if not (minlong < plong < maxlong):
            continue
        if not (minlat < plat < maxlat):
            continue
        x,y = transform(plong,plat,minlong,maxlong,minlat,maxlat)
        try:
            count_trips[(x,y)] += 1 
        except:
            count_trips[(x,y)] = 1
        if passcount == 1:
            try:
                count_1[(x,y)] += 1 
            except:
                count_1[(x,y)] = 1
        if passcount == 3:
            try:
                count_3[(x,y)] += 1 
            except:
                count_3[(x,y)] = 1
    return count_trips, count_1, count_3