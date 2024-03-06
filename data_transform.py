# map ip address from Fraud_Data.csv to country name from IpAddress_to_Country.csv
# save the result to EFraud_Data_Country.csv
import os
import pandas as pd
from intervaltree import IntervalTree


# read data_fraud from csv file
def read_data_fraud(filename):
    filename = os.path.join(os.path.dirname(__file__), 'data', filename)
    data = pd.read_csv(filename, parse_dates=['signup_time', 'purchase_time'],
                       converters={'ip_address': lambda x: preprocess_ip_to_int(x)})
    return data.copy()
# Read data_ip from csv file
def load_ip(filename):

    filename = os.path.join(os.path.dirname(__file__), 'data', filename)
    ip = pd.read_csv(filename, sep=';')
    return ip.copy()
# replace ip address with country name
def replace_Ip_country(data_fraud, data_ip):
    '''
    Ip address to country name transformation logarithm time complexity
    :param data_fraud: Fraud_Data.csv
    :param data_ip: IpAddress_to_Country.csv
    :return: transformed data_fraud
    '''
    search_tree = IntervalTree()
    for row in data_ip.itertuples(index=False):
        search_tree[row.lower_bound_ip_address:row.upper_bound_ip_address] = row.country

    def ip_lookup(ip):
        result = search_tree[ip]
        return result.pop().data if result else 'Not Found'

    data_fraud['country'] = data_fraud['ip_address'].apply(ip_lookup)

    return data_fraud
# convert ip address to int
def preprocess_ip_to_int(ip):
    return int(ip.split('.')[0])
# save data to csv file
def save_data(data, filename):
    filename = os.path.join(os.path.dirname(__file__), 'data', filename)
    data.to_csv(filename, index=False)

if __name__ == '__main__':
    data_fraud = read_data_fraud('Fraud_Data.csv')
    data_ip = load_ip('IpAddress_to_Country.csv')
    data_fraud = replace_Ip_country(data_fraud, data_ip)
    save_data(data_fraud, 'EFraud_Data_Country.csv')

