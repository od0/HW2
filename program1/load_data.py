import csv as csv
import numpy as np

def A(infile):
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