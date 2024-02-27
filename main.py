import os

import numpy as np
import pandas as pd


# load EFraud_Dataset with country
# Read data from csv files
def read_data(filename, delimiter=','):
    """
    Read the fraud data from csv file
    :param filename: csv file
    :return: dataframe
    """
    # Get the path of the file
    filename = os.path.join(os.path.dirname(__file__), 'data', filename)
    parse_dates = ['signup_time', 'purchase_time']
    data = pd.read_csv(filename, parse_dates=parse_dates, delimiter=delimiter)
    return data.copy()
# Save the text output to a file
def save_to_file(filename, text):
    with open(filename, 'a') as file:
        file.write(text)

# Check the dublicates value in columns and rows
def dublicates_values(data):
    text_output = '##################################\n'
    text_output += 'Missing Values'
    text_output += '##################################\n'
    duplicate_columns = data.columns.duplicated().sum()
    duplicate_rows = data.duplicated(subset='user_id').sum()
    text_output += 'Duplicate Columns:'
    text_output += str(duplicate_columns) + '\n'
    text_output += 'Duplicate Rows:'
    text_output += str(duplicate_rows)+ '\n'
    save_to_file('report.txt', text_output)
# check the timestamp if purchase time is after signup time
def time_difference(data):
    # check if signup time is before purchase time
    data['time_difference'] = (data['purchase_time'] - data['signup_time']).dt.total_seconds()
    # check if time difference is less than 0
    time_difference = data[data['time_difference'] < 0].shape[0]
    text_output = '##################################\n'
    text_output += 'Time Difference'
    text_output += '##################################\n'
    text_output += 'Time Difference:'
    text_output += str(time_difference) + '\n'
    save_to_file('report.txt', text_output)

# Tukey’s method for finding outliers
def tukey_method(data, column):
    text_output = '##################################\n'
    text_output += 'Tukey Method\n'
    text_output += '##################################\n'
    text_output += 'Column:'
    text_output += column + '\n'
    text_output += 'Outliers:'
    text_output += '\n'

    q75, q25 = np.percentile(data[column], [75, 25])
    iqr = q75 - q25
    lower_bound = q25 - (iqr * 1.5)
    upper_bound = q75 + (iqr * 1.5)
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)].shape[0]
    text_output += str(outliers) + '\n'
    save_to_file('report.txt', text_output)



if __name__ == '__main__':
    data = read_data('EFraud_Data_Country.csv')
    #dublicates_values(data)
    #time_difference(data)
    # tukey_method(data, 'purchase_value')
    # tukey_method(data, 'age')
############ Feature Engineering ################



