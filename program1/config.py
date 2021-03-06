__author__ = 'aub3'
"""
All constants should go here.
"""
bins = 20
minlong = -75
maxlong = -70
minlat = 40
maxlat = 41

EXAMPLE_DATA = "data/example_data.csv" # small data files (ideally < 5 MB) should be stored in data folder
TRIP_DATA_1 = "../../data/trip_data_1.csv" # Large data files can be outside your directory structure
TRIP_DATA_2 = "../../data/trip_data_2.csv" # Large data files can be outside your directory structure
TRAIN_DATA = "../../data/train_data.csv" # contains every 20th trip from trip_data_2.csv
TEST_DATA = "../../data/test_data.csv" # contains first 100,000 trips from trip_data_1.csv

FILE_FORMAT ={
    '0':'medallion',
    '1':'hack_license',
    '2':'vendor_id',
    '3':'rate_code',
    '4':'store_and_fwd_flag',
    '5':'pickup_datetime',
    '6':'dropoff_datetime',
    '7':'passenger_count',
    '8':'trip_time_in_secs',
    '9':'trip_distance',
    '10':'pickup_longitude',
    '11':'pickup_latitude',
    '12':'dropoff_longitude',
    '13':'dropoff_latitude'
}

F_FIELDS  = [8,9,10,11,12,13] # Float fields
S_FIELDS = [5,] # String fields