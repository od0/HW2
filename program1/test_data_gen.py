__author__ = 'aub3'
"""
Generates test_data.csv from trip_data_1.csv by selecting the first 100,000 trips
"""
from config import TRIP_DATA_1,TEST_DATA

if __name__ == '__main__':
    fout = open(TEST_DATA,'w')
    count = 0
    for i,line in enumerate(file(TRIP_DATA_1)):
        fout.write(line)
        if count == 100000:
        	break
        count += 1
    fout.close()
    print i,count