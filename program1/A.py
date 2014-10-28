import numpy as np
from scipy import stats
import csv as csv
from collections import defaultdict
import matplotlib.pyplot as plt

import config

def transform(plong,plat):
    return round((plong-config.minlong)*config.bins/(config.maxlong-config.minlong),0),round((plat-config.minlat)*config.bins/(config.maxlat-config.minlat),0)

def load_data(infile):
    count_trips = defaultdict(int)
    count_1 = defaultdict(int)
    count_3 = defaultdict(int)
    input = csv.reader(open(infile,'rb'))
    header = input.next()
    testbreak = 0
    for row in input:
        data = np.array(row)
        passcount = float(data[7])
        plong = float(data[10])
        plat = float(data[11])
        if not (config.minlong < plong < config.maxlong):
            continue
        if not (config.minlat < plat < config.maxlat):
            continue
        x,y = transform(plong,plat)
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

def plot_density(count_trips,count,title):
    grid = np.zeros((config.bins,config.bins))
    for (i,j),z in np.ndenumerate(grid):
        try:
            grid[j,i] = float(count[(i,j)]) / float(count_trips[(i,j)])
        except:
            grid[j,i] = 0
        #print "----"
        #print grid[i,j], i, j
        #print count[(i,j)]
        #print count_trips[(i,j)]
    grid = np.flipud(grid) #to counter matshow vertical flip
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(grid, cmap='spectral')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    xticks = np.linspace(config.minlong,config.maxlong,num=round(config.bins/2))
    yticks = np.linspace(config.minlat,config.maxlat,num=round(config.bins/2))
    yticks = yticks[::-1]
    xticks = np.around(xticks,decimals=1)
    yticks = np.around(yticks,decimals=1)
    xspace = np.linspace(0,config.bins-1,config.bins/2)
    yspace = np.linspace(0,config.bins-1,config.bins/2)
    plt.xticks(xspace,xticks)
    plt.yticks(yspace,yticks)
    for (i,j),z in np.ndenumerate(grid):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    plt.title(title)
    plt.show()



def main():
    count_trips, count_1, count_3 = load_data(config.TRAIN_DATA)
    #print count_1
    #print count_trips
    plot_density(count_trips, count_1,'Density Estimation for Single Passenger')
    plot_density(count_trips, count_3,'Density Estimation for Three Passengers')

if __name__ == '__main__':
    main()