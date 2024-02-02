import os
import sys

import pandas as pd


# Save the text output to a file
def save_to_file(filename, text):
    with open(filename, 'a') as file:
        file.write(text)


# Read data from csv files
def read_data(filename, parse_dates=None, delimiter=','):
    """
    Read the fraud data from csv file
    :param filename: csv file
    :return: dataframe
    """
    # Get the path of the file
    filename = os.path.join(os.path.dirname(__file__), 'data', filename)
    # Read the data from csv file
    data = pd.read_csv(filename, parse_dates=parse_dates, delimiter=delimiter)
    return data.copy()


# Exploring data frames
def explore_data_frames(data):
    """
    Read the 4 first rows of the dataframe
    :param data: dataframe
    :return: None
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('expand_frame_repr', False)
    text_output = '##############################################\n'
    text_output += '############ Exploring data frames ###########\n'
    text_output += '##############################################\n'
    text_output += str(data.head()) + '\n'
    save_to_file('report.txt', text_output)


# Identifying missing values
def missing_values(data):
    """
    Get the number of missing values
    :param data: data frame
    :return: None
    """
    text_output = '##############################################\n'
    text_output += '############# Missing values #################\n'
    text_output += '##############################################\n'
    text_output += str(data.isnull().sum()) + '\n'
    save_to_file('report.txt', text_output)


# Check the data if has null values for each column
def check_null_values(data):
    """
    Check if the data has null values for each column
    :param data: data frame
    :return: None
    """
    text_output = '##############################################\n'
    text_output += '########### Check null values ################\n'
    text_output += '##############################################\n'
    text_output += str(data.isnull().any()) + '\n'
    save_to_file('report.txt', text_output)


# datatframe data types
def data_types(data):
    """
    Get the data types of each column
    :param data: data frame
    :return: None
    """
    text_output = '##############################################\n'
    text_output += '############ Data types ######################\n'
    text_output += '##############################################\n'
    # Map pandas data types to Python types
    python_types = {
        'int64': 'int',
        'float64': 'float',
        'object': 'str',
        'bool': 'bool',
        'datetime64[ns]': 'datetime',
    }

    # Use apply to map data types for each column
    data_types_mapped = data.apply(lambda x: python_types.get(str(x.dtypes), 'Unknown')).rename('Data types')
    dtypes_df = pd.DataFrame(data_types_mapped)

    text_output += str(dtypes_df) + '\n'

    save_to_file('report.txt', text_output)


# Data Frames attributes
def data_metadata(data):
    """
    Get the data metadata
    :param data: data frame
    :return: None
    """
    text_output = '##############################################\n'
    text_output += '################ Data metadata ###############\n'
    text_output += '##############################################\n'
    # Get the data frames attributes
    metadatas = ['index', 'shape', 'size', 'ndim', 'empty']
    for metadata in metadatas:
        text_output += 'Data frame attribute: ' + metadata + '\n'
        text_output += str(getattr(data, metadata)) + '\n'
    save_to_file('report.txt', text_output)


# Descriptive Statistics for numeric columns
def descriptive_statistics_numerical(data):
    """
    Get the descriptive statistics for numeric columns
    :param data: data frame
    :return: None
    """
    pd.options.display.float_format = '{:,.2f}'.format
    # data numerical columns purchase_value, age, ip_address
    data = data[['purchase_value', 'age', 'ip_address']]
    text_output = '##############################################\n'
    text_output += '### Descriptive statistics Numerical data ####\n'
    text_output += '##############################################\n'
    text_output += str(data.describe()) + '\n'
    save_to_file('report.txt', text_output)


# Check if user_id is is assigned sequentially
def check_user_id(data):
    data_sorted = data.sort_values(by=['user_id'])
    is_sequential = data_sorted['user_id'].is_monotonic_increasing
    text_output = '##############################################\n'
    text_output += '########### Check user_id sequence ###########\n'
    text_output += '##############################################\n'
    text_output += 'Is user_id sequential: ' + str(is_sequential) + '\n'
    save_to_file('report.txt', text_output)

#Check how many users ids are not unique
def check_unique_user_id(data):
    text_output = '##############################################\n'
    text_output += 'Check users made more then one transanction \n'
    text_output += '##############################################\n'
    text_output += 'Number of user more one transactions: ' + str((len(data) - data['user_id'].nunique())) + '\n'
    save_to_file('report.txt', text_output)

# Get the descriptive statistics for categorical columns
def descriptive_statistics_categorical(data):
    """
    Get the descriptive statistics for categorical columns
    :param data: data frame
    :return: None
    """
    text_output = '##############################################\n'
    text_output += '## Descriptive statistics categorical data ###\n'
    text_output += '##############################################\n'
    # data categorical columns device_id, source, browser, sex, user_id
    data = data[['device_id', 'source', 'browser', 'sex', 'user_id', 'class']].copy()
    # transform user_id  as categorical data
    data['user_id'] = data['user_id'].astype('category')
    data['class'] = data['class'].astype('category')
    text_output += str(data.describe()) + '\n'
    save_to_file('report.txt', text_output)


# Check all date from datetime columns have format='%Y-%m-%d %H:%M:%S'
def check_datetime_format(data):
    """
    Check all date from datetime columns have format='%Y-%m-%d %H:%M:%S'
    :param data: data frame
    :return: None
    """
    text_output = '##############################################\n'
    text_output += '########### Check datetime format ############\n'
    text_output += '##############################################\n'
    datetime_columns = ['signup_time', 'purchase_time']
    for datetime_column in datetime_columns:
        try:
            pd.to_datetime(data[datetime_column], format='%Y-%m-%d %H:%M:%S')
            text_output += 'All dates from ' + datetime_column + ' have format=%Y-%m-%d %H:%M:%S\n'
        except ValueError:
            text_output += 'Not all dates from ' + datetime_column + ' have format=%Y-%m-%d %H:%M:%S\n'
    save_to_file('report.txt', text_output)


# Date time analysis
def descriptive_statistics_datetime(data):
    """
    Get the descriptive statistics for datetime columns
    :param data: data frame
    :return: None
    """
    text_output = '##############################################\n'
    text_output += '## Descriptive statistics datetime data #####\n'
    text_output += '##############################################\n'
    # Get min and max of datetime columns
    datetime_columns = ['signup_time', 'purchase_time']
    for datetime_column in datetime_columns:
        text_output += 'Earliest date ' + datetime_column + ': ' + str(data[datetime_column].min()) + '\n'
        text_output += 'Latest Date ' + datetime_column + ': ' + str(data[datetime_column].max()) + '\n'
    # Get the month with the most transactions
    data['month'] = data['purchase_time'].dt.month
    text_output += 'Month with the most transactions: ' + str(data['month'].mode()[0]) + '\n'
    # Get the day of the week with the most transactions
    data['day'] = data['purchase_time'].dt.day
    text_output += 'Day with the most transactions: ' + str(data['day'].mode()[0]) + '\n'
    # Get the hour with the most transactions
    data['hour'] = data['purchase_time'].dt.hour
    text_output += 'Hour with the most transactions: ' + str(data['hour'].mode()[0]) + '\n'
    save_to_file('report.txt', text_output)


# Check IpAddress_to_Country.csv data type and convert for Pandas to Python types
def check_ip_address(data):
    """
    Check IpAddress_to_Country.csv data type and convert for Pandas to Python types
    :param data: data frame
    :return: None
    """
    text_output = '##############################################\n'
    text_output += '########### Check Ip Address data type ######\n'
    text_output += '##############################################\n'
    # Map pandas data types to Python types
    python_types = {
        'int64': 'int',
        'float64': 'float',
        'object': 'str',
        'bool': 'bool',
        'datetime64[ns]': 'datetime',
    }
    # Use apply to map data types for each column
    data_types_mapped = data.apply(lambda x: python_types.get(str(x.dtypes), 'Unknown')).rename('Data types')
    dtypes_df = pd.DataFrame(data_types_mapped)
    text_output += str(dtypes_df) + '\n'
    save_to_file('report.txt', text_output)
# Check missing values
def check_missing_values(data):
    """
    Get the number of missing values
    :param data: data frame
    :return: None
    """
    text_output = '##############################################\n'
    text_output += '############# Missing values #################\n'
    text_output += '##############################################\n'
    text_output += str(data.isnull().sum()) + '\n'
    save_to_file('report.txt', text_output)
# Descriptive Statistics for numerical columns IpAddress_to_Country.csv
def descriptive_statistics_numerical_ip_address(data):
    """
    Get the descriptive statistics for numeric columns
    :param data: data frame
    :return: None
    """
    pd.options.display.float_format = '{:,.2f}'.format
    # Remove from data datatime columns
    data = data.select_dtypes(include=['int64'])
    text_output = '##############################################\n'
    text_output += '# Descriptive statistics Ip address columns #\n'
    text_output += '##############################################\n'
    text_output += str(data.describe()) + '\n'
    save_to_file('report.txt', text_output)
# Descriptive Statistics for categorical columns IpAddress_to_Country.csv
def descriptive_statistics_categorical_ip_address(data):
    """
    Get the descriptive statistics for categorical columns
    :param data: data frame
    :return: None
    """
    text_output = '##############################################\n'
    text_output += '## Descriptive statistics Ip-Country data ###\n'
    text_output += '##############################################\n'
    text_output += str(data.describe(include='object')) + '\n'
    save_to_file('report.txt', text_output)

# Descriptive Statistics for country column
def descriptive_statistics_country(data):
    """
    Get the descriptive statistics for categorical columns
    :param data: data frame
    :return: None
    """
    text_output = '##############################################\n'
    text_output += '## Descriptive statistics country data ###\n'
    text_output += '##############################################\n'
    text_output += str(data['country'].describe()) + '\n'
    save_to_file('report.txt', text_output)

# Find the total amount from country column 'Not Found' values
def find_not_found_country(data):
    """
    Find the total amount from country column 'Not Found' values
    :param data: data frame
    :return: None
    """
    text_output = '##############################################\n'
    text_output += '## Total amount country Not Found ##########\n'
    text_output += '##############################################\n'
    text_output += str(data['country'].value_counts()['Not Found']) + '\n'
    save_to_file('report.txt', text_output)

# python init main
if __name__ == '__main__':
    # read fraud data
    data_fraud = read_data('Fraud_Data.csv', parse_dates=['signup_time', 'purchase_time'])
    data_ip = read_data('IpAddress_to_Country.csv', delimiter=';')
######## Fraud_Data.csv ###########
    # Exploring data frames
    #explore_data_frames(data_fraud)
    # Missing values
    #missing_values(data_fraud)
    # Check null values
    #check_null_values(data_fraud)
    # Datatframe data types
    #data_types(data_fraud)
    # Data metadata
    #data_metadata(data_fraud)
    # Descriptive statistics numerical
    #descriptive_statistics_numerical(data_fraud)
    # Check user_id sequence
    #check_user_id(data_fraud)
    # Check unique user_id
    #check_unique_user_id(data_fraud)
    # Descriptive statistics categorical
    descriptive_statistics_categorical(data_fraud)
    # Check datetime format
    #check_datetime_format(data_fraud)
    # Descriptive statistics datetime
    #descriptive_statistics_datetime(data_fraud)
######## IpAddress_to_Country.csv ###########
    # Check Ip Address data type
    #check_ip_address(data_ip)
    # Check missing values
    #check_missing_values(data_ip)
    # Check null values
    #check_null_values(data_ip)
    # Descriptive statistics numerical
    #descriptive_statistics_numerical_ip_address(data_ip)
    # Descriptive statistics categorical
    #descriptive_statistics_categorical_ip_address(data_ip)
############# Data Transform ################
    # load transformed data
    #data_transformed = read_data('EFraud_Data_Country.csv', parse_dates=['signup_time', 'purchase_time'])
    # Descriptive statistics country
    #descriptive_statistics_country(data_transformed)
    # Find the total amount from country column 'Not Found' values
    #find_not_found_country(data_transformed)



