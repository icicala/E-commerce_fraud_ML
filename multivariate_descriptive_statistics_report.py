import pandas as pd
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
def save_to_file(filename, text):
    with open(filename, 'a') as file:
        file.write(text)

# load EFraud_Dataset with country
def load_efraud_dataset(filename):
    filename = os.path.join(os.path.dirname(__file__), 'data', filename)
    data = pd.read_csv(filename, parse_dates=['signup_time', 'purchase_time'])
    return data.copy()


# Identify the relationship between features(user_id, purchase_value, age) and label(class) column
def relationship_between_numerical_features_and_label(data):
    text_output = '##############################################\n'
    text_output += '######## Relation numerical features with label(class) ########\n'
    text_output += '##############################################\n'
    numerical_columns = data[['user_id', 'purchase_value', 'age', 'class']]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.0f' % x)
    grouped_data = numerical_columns.groupby('class').describe()
    # Convert the DataFrameGroupBy object to a string
    text_output += str(grouped_data) + '\n'

    # Save the output to a file
    save_to_file('report.txt', text_output)


# Identify the relationship between datetime features(signup_time, purchase_time) and label(class) column
def relationship_between_datetime_features_and_label(data):
    text_output = '##############################################\n'
    text_output += '######## Relation datetime features with label(class) ########\n'
    text_output += '##############################################\n'
    datetime_columns = data[['signup_time', 'purchase_time', 'class']]
    pd.set_option('display.max_columns', None)
    grouped_data = datetime_columns.groupby('class').describe()
    # Convert the DataFrameGroupBy object to a string
    text_output += str(grouped_data) + '\n'

    # Save the output to a file
    save_to_file('report.txt', text_output)


# Identify the relationship between categorical features: sex, source, browser, device_id, country) and label(class) column
def relationship_between_categorical_features_and_label(data):
    text_output = '##############################################\n'
    text_output += '##### Relation categorical features with label(class) ####\n'
    text_output += '##############################################\n'
    categorical_features = data[['sex', 'source', 'browser', 'device_id', 'country', 'class']]
    pd.set_option('display.max_columns', None)
    grouped_data = categorical_features.groupby('class').describe()
    # Convert the DataFrameGroupBy object to a string
    text_output += str(grouped_data) + '\n'

    # Save the output to a file
    save_to_file('report.txt', text_output)

# Relation of catgorical features with categorical features: sex, source, browser, device_id, country
def relation_between_categorical_features_and_country(data):
    text_output = '##############################################\n'
    text_output += '##### Relation categorical features with categorical features ####\n'
    text_output += '##############################################\n'

    # Filter data for fraud class (class == 1)
    fraud_data = data[data['class'] == 1]

    # Select relevant columns
    categorical_features = fraud_data[['sex', 'source', 'browser', 'device_id', 'country']]
    pd.set_option('display.max_columns', None)
    # Get the top 15 countries
    top_countries = categorical_features['country'].value_counts().nlargest(15).index.tolist()

    # Filter data for top 15 countries
    categorical_features_top15 = categorical_features[categorical_features['country'].isin(top_countries)]

    # Group by country and describe
    grouped_data = categorical_features_top15.groupby('country').describe()



    # Convert the DataFrameGroupBy object to a string
    text_output += str(grouped_data) + '\n'

    # Save the output to a file
    save_to_file('report.txt', text_output)

def relation_between_categorical_features_and_category(data):
    text_output = '##############################################\n'
    text_output += '##### Relation categorical features with source browser sex device_id ####\n'
    text_output += '##############################################\n'

    fraud_data = data[data['class'] == 1]
    categorical_features = fraud_data[['source', 'browser', 'sex', 'device_id']]
    pd.set_option('display.max_columns', None)
    grouped_data = categorical_features.groupby(['sex', 'source', 'browser']).describe(include='all')
    # Convert the DataFrameGroupBy object to a string
    text_output += str(grouped_data) + '\n'

    # Save the output to a file
    save_to_file('report.txt', text_output)


# Relationship between categorical:source and numerical:age
def relation_source_age(data):
    text_output = '##############################################\n'
    text_output += '##### Source and age association ####\n'
    text_output += '##############################################\n'

    # Filter data for fraud class (class == 1)
    fraud_data = data[data['class'] == 1]
    pd.set_option('display.max_columns', None)
    data_tab = pd.crosstab(index=fraud_data['source'], columns=fraud_data['age'])

    # Convert the DataFrame to a string
    text_output += str(data_tab) + '\n'

    # Save the output to a file
    save_to_file('report.txt', text_output)

# Count the number of users for each device_id
def count_users_with_multiple_devices(data):
    text_output = '##############################################\n'
    text_output += '##### Count the number of users for each device_id ####\n'
    text_output += '##############################################\n'

    # Filter data for fraud class (class == 1)
    fraud_data = data[data['class'] == 1]
    pd.set_option('display.max_columns', None)
    data_tab = fraud_data.groupby('device_id')['user_id'].nunique()
    devices_multiple_users = data_tab[data_tab > 1]
    # save the total number of devices with multiple users
    text_output += 'Total number of devices with multiple users: ' + str(len(devices_multiple_users)) + '\n'
    # save the the maximum number of users for a device
    text_output += 'Maximum number of users for a device: ' + str(devices_multiple_users.max()) + '\n'
    # save the the minimum number of users for a device
    text_output += 'Minimum number of users for a device: ' + str(devices_multiple_users.min()) + '\n'
    # sort in accending order the number of users for a device
    devices_multiple_users = devices_multiple_users.sort_values(ascending=True)
    # Convert the DataFrame to a string
    text_output += str(devices_multiple_users) + '\n'

    # Save the output to a file
    save_to_file('report.txt', text_output)

# Relation between categorical features and datetime features: signup_time, purchase_time
def relation_categorical_datetime(data):
    text_output = '##############################################\n'
    text_output += '##### Relation categorical features with datetime features ####\n'
    text_output += '##############################################\n'
    fraud_data = data[data['class'] == 1]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.0f' % x)
    # display all the raws
    pd.set_option('display.max_rows', None)
    column_features = fraud_data[['device_id', 'country', 'source', 'purchase_time']]
    grouped_data = column_features.groupby(['source', 'country']).agg({'purchase_time': ['count', 'mean', 'min', 'max']})
    # sort the data by count
    grouped_data = grouped_data.sort_values(by=[('purchase_time', 'count')], ascending=False)
    # Convert the DataFrameGroupBy object to a string
    text_output += str(grouped_data) + '\n'

    # Save the output to a file
    save_to_file('report.txt', text_output)


# cross tabulation between categorical features and label
def cross_tabulation_categorical_features_and_label(data):
    text_output = '##############################################\n'
    text_output += '##### Cross tabulation categorical features with label ####\n'
    text_output += '##############################################\n'
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.0f' % x)
    # display all the raws
    pd.set_option('display.max_rows', None)
    column_features = data[['sex', 'browser', 'source']]
    # cross tabulation between categorical features and label
    for cat_column in column_features:
        data_tab = pd.crosstab(index=data[cat_column], columns=data['class'])
        text_output += str(data_tab) + '\n'

    # Save the output to a file
    save_to_file('report.txt', text_output)

#anova test categorical-numerical variables
def anova_categorical_numerical(data):
    text_output = '##############################################\n'
    text_output += '##### Anova test categorical-numerical variables ####\n'
    text_output += '##############################################\n'

    # Categorical features
    categorical_features = ['source', 'browser', 'country']

    # Numerical features
    numerical_features = ['purchase_value', 'age', 'user_id']

    # Loop through the categorical features
    for categorical_feature in categorical_features:
        # Loop through the numerical features
        for numerical_feature in numerical_features:
            # Perform ANOVA test
            formula = f'{numerical_feature} ~ C({categorical_feature})'
            anova = ols(formula, data=data).fit()
            anova_table = anova_lm(anova, typ=2)
            # save the categorical feature and numerical feature and anova table
            text_output += 'Categorical feature: ' + categorical_feature + '\n'
            text_output += 'Numerical feature: ' + numerical_feature + '\n'
            text_output += str(anova_table) + '\n'

    # Save the output to a file
    save_to_file('report.txt', text_output)


# descriptive statistics between country and purchase_value
def descriptive_statistics_country_purchase_value(data):
    text_output = '##############################################\n'
    text_output += '##### Descriptive statistics between country and purchase_value ####\n'
    text_output += '##############################################\n'
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.0f' % x)
    # display all the raws
    pd.set_option('display.max_rows', None)
    # filter data for fraud class (class == 1)
    fdata = data[data['class'] == 0]
    column_features = fdata[['country', 'purchase_value']]
    grouped_data = column_features.groupby(['country']).agg({'purchase_value': ['count', 'mean', 'min', 'max']})
    # sort the data by count
    grouped_data = grouped_data.sort_values(by=[('purchase_value', 'count')], ascending=False)
    # Convert the DataFrameGroupBy object to a string
    text_output += str(grouped_data) + '\n'

    # Save the output to a file
    save_to_file('report.txt', text_output)


#initialize the python script
if __name__ == '__main__':
    # load dataset
    data = load_efraud_dataset('Efraud_Data_Country.csv')
    # relationship between features and label
    #relationship_between_numerical_features_and_label()
    # relationship between datetime features and label
    #relationship_between_datetime_features_and_label(data)
    # relationship between categorical features and label
    #relationship_between_categorical_features_and_label(data)
    # relationship between categorical features and categorical features
    #relation_between_categorical_features_and_country(data)
    #relation_between_categorical_features_and_category(data)
    # relation between source and age
    #relation_source_age(data)
    # count users by device_id
    #count_users_with_multiple_devices(data)
    # relation between categorical features and datetime features
    #relation_categorical_datetime(data)
    # cross tabulation between categorical features and label
    #cross_tabulation_categorical_features_and_label(data)
    # Cramer's V statistic for categorical variable and class
    # anova test categorical-numerical variables
    #anova_categorical_numerical(data)
    descriptive_statistics_country_purchase_value(data)


