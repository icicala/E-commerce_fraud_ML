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

# feature creation the number of ip address per user
def feature_creation_ip_address(data):
    data['number_users_perIP'] = data.groupby('ip_address')['user_id'].transform('count')
    nan_values = data['number_users_perIP'].isna().sum()
    text_output = '##################################'
    text_output += 'Feature Creation'
    text_output += '##################################\n'
    text_output += 'Number of users per IP:'
    text_output += str(nan_values) + '\n'
    save_to_file('report.txt', text_output)
    return data
# feature creation the number of signup_time per device id
def feature_creation_signup_time(data):
    data['signup_count_PerDevice'] = data.groupby('device_id')['signup_time'].transform('count')
    nan_values = data['signup_count_PerDevice'].isna().sum()
    text_output = '##################################'
    text_output += 'Feature Creation'
    text_output += '##################################\n'
    text_output += 'Number of signups per device:'
    text_output += str(nan_values) + '\n'
    save_to_file('report.txt', text_output)
    return data
# feature creation between purchase time and signup time
def feature_creation_time_difference(data):
    data['time_difference'] = (data['purchase_time'] - data['signup_time']).dt.total_seconds()
    data['time_difference'] = data['time_difference'].astype(int)
    nan_values = data['time_difference'].isna().sum()
    text_output = '##################################'
    text_output += 'Feature Creation'
    text_output += '##################################\n'
    text_output += 'Time Difference:'
    text_output += str(nan_values) + '\n'
    save_to_file('report.txt', text_output)
    return data
# feature creation extract from signup time week number, day number of month, day number of week, hour(mm and ss) in seconds
def feature_extraction_signup_time(data):
    data['signup_week'] = data['signup_time'].dt.isocalendar().week
    data['signup_day'] = data['signup_time'].dt.day
    data['signup_day_of_week'] = data['signup_time'].dt.dayofweek
    data['signup_time_seconds'] = data['signup_time'].dt.hour * 3600 + data['signup_time'].dt.minute * 60 + data['signup_time'].dt.second
    nan_values = data[['signup_week', 'signup_day', 'signup_day_of_week', 'signup_time_seconds']].isna().sum()
    text_output = '##################################'
    text_output += 'Feature Creation'
    text_output += '##################################\n'
    text_output += 'Signup TIme:\n'
    text_output += str(nan_values['signup_week']) + '\n'
    text_output += str(nan_values['signup_day']) + '\n'
    text_output += str(nan_values['signup_day_of_week']) + '\n'
    text_output += str(nan_values['signup_time_seconds']) + '\n'
    save_to_file('report.txt', text_output)
    return data
# feature creation extract from purchase time week number, day number of month, day number of week, hour(mm and ss) in seconds
def feature_extraction_purchase_time(data):
    data['purchase_week'] = data['purchase_time'].dt.isocalendar().week
    data['purchase_day'] = data['purchase_time'].dt.day
    data['purchase_day_of_week'] = data['purchase_time'].dt.dayofweek
    data['purchase_time_seconds'] = data['purchase_time'].dt.hour * 3600 + data['purchase_time'].dt.minute * 60 + data['purchase_time'].dt.second
    nan_values = data[['purchase_week', 'purchase_day', 'purchase_day_of_week', 'purchase_time_seconds']].isna().sum()
    text_output = '##################################'
    text_output += 'Feature Creation'
    text_output += '##################################\n'
    text_output += 'Purchase Time:\n'
    text_output += str(nan_values['purchase_week']) + '\n'
    text_output += str(nan_values['purchase_day']) + '\n'
    text_output += str(nan_values['purchase_day_of_week']) + '\n'
    text_output += str(nan_values['purchase_time_seconds']) + '\n'
    save_to_file('report.txt', text_output)
    return data
# feature creation count the number of user per device id
def feature_creation_user_per_device(data):
    data['number_user_per_device'] = data.groupby('device_id')['user_id'].transform('count')
    nan_values = data['number_user_per_device'].isna().sum()
    text_output = '##################################'
    text_output += 'Feature Creation'
    text_output += '##################################\n'
    text_output += 'User per device:\n'
    text_output += str(nan_values) + '\n'
    save_to_file('report.txt', text_output)
    return data
# feature creation count the number of user per country
def feature_creation_user_per_country(data):
    data['number_user_per_country'] = data.groupby('country')['user_id'].transform('count')
    nan_values = data['number_user_per_country'].isna().sum()
    text_output = '##################################'
    text_output += 'Feature Creation'
    text_output += '##################################\n'
    text_output += 'User per country:\n'
    text_output += str(nan_values) + '\n'
    save_to_file('report.txt', text_output)
    return data
if __name__ == '__main__':
    data = read_data('EFraud_Data_Country.csv')
    #dublicates_values(data)
    #time_difference(data)
    # tukey_method(data, 'purchase_value')
    # tukey_method(data, 'age')
############ Feature Engineering ################
    data = feature_creation_ip_address(data)
    data = feature_creation_signup_time(data)
    data = feature_creation_time_difference(data)
    data = feature_extraction_signup_time(data)
    data = feature_extraction_purchase_time(data)
    data = feature_creation_user_per_device(data)
    data = feature_creation_user_per_country(data)
    print(data.head())






