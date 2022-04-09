from pandas import read_csv
from process import ProcessDateset


class Shealth:

    raw_data = read_csv('shealth.csv',  sep=";", decimal=',', parse_dates=['Date'])
    process = ProcessDateset()
    data_set = process.process_data(raw_data)

    data_set_description = data_set.describe()

