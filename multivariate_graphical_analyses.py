import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.stats import pointbiserialr
from scipy.stats.contingency import association
import scipy.stats as stats

from ydata_profiling import ProfileReport
from pandas.plotting import autocorrelation_plot
from scipy import fft
from scipy.stats import spearmanr


# Function to save plots as PNG files
def save_plot_as_png(plot_function, plot_name):
    # save to folder 'plots'
    os.makedirs('plots', exist_ok=True)
    # Plotting
    plot_function()
    # Save plot as PNG file
    plt.savefig(os.path.join('plots', plot_name + '.png'))
    # Clear plot
    plt.clf()


def load_efraud_dataset(filename):
    filename = os.path.join(os.path.dirname(__file__), 'data', filename)
    data = pd.read_csv(filename, parse_dates=['signup_time', 'purchase_time'])
    return data.copy()

# Pandas data profiling report
def data_profiling_report(data):
    # Create a pandas profiling report
    profile = ProfileReport(data, title='Pandas Profiling Report', explorative=True)
    # Save the report as HTML file
    profile.to_file(os.path.join('reports', 'pandas_profiling_report.html'))

# Count plot of class and discrete numerical features using relative frequency
def relationship_between_numerical_features_and_label(data):
    def plot_age():
        # custom palette
        custom_palette = ['grey', 'coral']
        # count plot of age and class
        ax = sns.countplot(data=data, x='age', hue='class', stat='proportion', palette=custom_palette)
        # Set the title
        plt.title('Count Plot of Age and Class')
        # Set x-axis label
        plt.xlabel('Age')
        # Set y-axis label
        plt.ylabel('Relative Frequency')
        # Show the legend with class meaning 0: Not Fraud, 1: Fraud
        plt.legend(title='Class', loc='upper right', labels=['Not Fraud', 'Fraud'])
        # fix the problem cluttering the x-axis by showing every every second age
        for ind, label in enumerate(ax.get_xticklabels()):
            # descriptive statiscs of the age
            age_descriptive_stats = data['age'].describe()
            # remove all data from x-axis
            ax.xticks = []
            # show only the minimum, maximum and mean age, Q1 and Q3 on the x-axis
            if label.get_text() in [str(int(age_descriptive_stats['min'])), str(int(age_descriptive_stats['25%'])), str(int(age_descriptive_stats['75%'])), str(int(age_descriptive_stats['max'])), str(int(age_descriptive_stats['mean']))]:
                label.set_visible(True)
            else:
                label.set_visible(False)
        # point-biserial correlation between age and class
        corr_coeff, p_value = pointbiserialr(data['age'], data['class'])
        # put the correlation coefficient and p-value on the plot
        plt.text(0.5, 0.5, 'Correlation Coefficient: %.2f\nP-value: %.2f' % (corr_coeff, p_value),
                    horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        # save the plot as png
    save_plot_as_png(plot_age, 'countplot_age')

    def plot_purchase_value():
        # custom palette
        custom_palette = ['grey', 'coral']
        # count plot of purchase_value and class
        ax = sns.countplot(data=data, x='purchase_value', hue='class', stat='proportion', palette=custom_palette)
        # Set the title
        plt.title('Count Plot of Purchase Value and Class')
        # Set x-axis label
        plt.xlabel('Purchase Value')
        # Set y-axis label
        plt.ylabel('Relative Frequency')
        # Show the legend with class meaning 0: Not Fraud, 1: Fraud
        plt.legend(title='Class', loc='upper right', labels=['Not Fraud', 'Fraud'])
        # fix the problem cluttering the x-axis by showing every every second purchase_value
        for ind, label in enumerate(ax.get_xticklabels()):
            # descriptive statiscs of the purchase_value
            purchase_value_descriptive_stats = data['purchase_value'].describe()
            # remove all data from x-axis
            ax.xticks = []
            # show only the minimum, maximum and mean purchase_value, Q1 and Q3 on the x-axis
            if label.get_text() in [str(int(purchase_value_descriptive_stats['min'])), str(int(purchase_value_descriptive_stats['25%'])), str(int(purchase_value_descriptive_stats['75%'])), str(int(purchase_value_descriptive_stats['max'])), str(int(purchase_value_descriptive_stats['mean']))]:
                label.set_visible(True)
            else:
                label.set_visible(False)
        # point-biserial correlation between purchase_value and class
        corr_coeff, p_value = pointbiserialr(data['purchase_value'], data['class'])
        # put the correlation coefficient and p-value on the plot
        plt.text(0.5, 0.5, 'Correlation Coefficient: %.2f\nP-value: %.2f' % (corr_coeff, p_value),
                    horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        # save the plot as png
    save_plot_as_png(plot_purchase_value, 'countplot_purchase_value')




# kernel density plot of correlation between datetime features(signup_time, purchase_time) and label(class) column
def relationship_between_datetime_features_and_label(data):
    def plot_signup_time():
        # Convert inf values to NaN before operating
        data['signup_time'] = data['signup_time'].replace([np.inf, -np.inf], np.nan)
        # Set up the figure
        plt.figure(figsize=(12, 8))
        # custom palette
        custom_palette = ['black', 'coral']
        # Create ridgeline plot
        sns.kdeplot(data=data, x='signup_time', hue='class', fill=True, common_norm=True, alpha=0.3, palette=custom_palette)
        # Set the title
        plt.title('Grouped Kernel Density Plot of Sign Up Time')
        # Set x-axis label
        plt.xlabel('Sign Up Time')
        # Set y-axis label
        plt.ylabel('Frequency')
        # change the 0 not fraud and 1 fraud
        plt.legend(title='Class', loc='upper right', labels=['Fraud', 'Not Fraud'])
    save_plot_as_png(plot_signup_time, 'kde_signup_time')

    def plot_purchase_time():
        # Convert inf values to NaN before operating
        data['purchase_time'] = data['purchase_time'].replace([np.inf, -np.inf], np.nan)
        plt.figure(figsize=(12, 8))
        # custom palette
        custom_palette = ['black', 'coral']
        # Create ridgeline plot
        sns.kdeplot(data=data, x='purchase_time', hue='class', fill=True, common_norm=True, alpha=0.3, palette=custom_palette)
        # Set the title
        plt.title('Grouped Kernel Density Plot of Purchase Time')
        # Set x-axis label
        plt.xlabel('Purchase Time')
        # Set y-axis label
        plt.ylabel('Frequency')
        # change the 0 not fraud and 1 fraud
        plt.legend(title='Class', loc='upper right', labels=['Fraud', 'Not Fraud'])
    save_plot_as_png(plot_purchase_time, 'kde_purchase_time')




# KDE sign up time day of the week and class
def plot_sign_up_dateweek_features(data):
    def plot_function():
        datetime_columns = data[['signup_time', 'purchase_time', 'class']].copy()
        pd.set_option('display.max_columns', None)
        # Extract day of the week component
        datetime_columns['signup_day_of_week'] = datetime_columns['signup_time'].dt.dayofweek
        # Set up the figure
        plt.figure(figsize=(12, 8))
        # Create KDE plot for the day of the week component
        sns.kdeplot(data=datetime_columns, x='signup_day_of_week', hue='class', fill=True, common_norm=False, alpha=0.3)
        # legent to show class meaning 0: Not Fraud, 1: Fraud
        plt.legend(title='Class', loc='upper right', labels=['Fraud', 'Not Fraud'])
        # legend to show xaxis meaning 0: Monday, 1: Tuesday, 2: Wednesday, 3: Thursday, 4: Friday, 5: Saturday, 6: Sunday
        plt.xticks(np.arange(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        # set the x axis from 0 to 6
        plt.xlim(0, 6)
        # Set the title
        plt.title('KDE Plot of Sign Up Day of the Week')
        # Set x-axis label
        plt.xlabel('Sign Up Day of the Week')
        # Set y-axis label
        plt.ylabel('Frequency')
        # point-biserial correlation between signup_day_of_week:0-Monday, 1-Tue and class: 0-Not Fraud, 1-Fraud
        corr_coeff, p_value = pointbiserialr(datetime_columns['signup_day_of_week'], datetime_columns['class'])
        # put the correlation coefficient and p-value on the plot
        plt.text(0.5, 0.5, 'Correlation Coefficient: %.2f\nP-value: %.2f' % (corr_coeff, p_value),
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    save_plot_as_png(plot_function, 'signup_kdeplot_day_of_week')


# kde plot hours of the day(signup_time) and class
def plot_sign_up_datehours_features(data):
    def plot_function():
        datetime_columns = data[['signup_time', 'purchase_time', 'class']].copy()
        pd.set_option('display.max_columns', None)
        # Extract time component (hour) from signup_time
        datetime_columns['signup_hour'] = datetime_columns['signup_time'].dt.hour
        # Set up the figure
        plt.figure(figsize=(12, 8))
        # Create KDE plot for the time component (hour)
        sns.kdeplot(data=datetime_columns, x='signup_hour', hue='class', fill=True, common_norm=False, alpha=0.3)
        # Legend to show class meaning 0: Not Fraud, 1: Fraud
        plt.legend(title='Class', loc='upper right', labels=['Fraud', 'Not Fraud'])
        # Set the x axis from 0 to 23
        plt.xlim(0, 23)
        # set the x axis all the hours of the day
        plt.xticks(np.arange(24))
        # Set the title
        plt.title('KDE Plot of Sign Up Hour of the Day')
        # Set x-axis label
        plt.xlabel('Sign Up Hour of the Day')
        # Set y-axis label
        plt.ylabel('Frequency')
        # point-biserial correlation between signup_hour and class
        corr_coeff, p_value = pointbiserialr(datetime_columns['signup_hour'], datetime_columns['class'])
        # put the correlation coefficient and p-value on the plot
        plt.text(0.5, 0.5, 'Correlation Coefficient: %.2f\nP-value: %.2f' % (corr_coeff, p_value),
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    save_plot_as_png(plot_function, 'signup_kdeplot_hour_of_day')


# KDE plot purchase time day of the week and class
def plot_purchase_dateweek_features(data):
    def plot_function():
        datetime_columns = data[['signup_time', 'purchase_time', 'class']].copy()
        pd.set_option('display.max_columns', None)
        # Extract day of the week component
        datetime_columns['purchase_day_of_week'] = datetime_columns['purchase_time'].dt.dayofweek
        # Set up the figure
        plt.figure(figsize=(12, 8))
        # Create KDE plot for the day of the week component
        sns.kdeplot(data=datetime_columns, x='purchase_day_of_week', hue='class', fill=True, common_norm=False,
                    alpha=0.3)
        # legent to show class meaning 0: Not Fraud, 1: Fraud
        plt.legend(title='Class', loc='upper right', labels=['Fraud', 'Not Fraud'])
        # legend to show xaxis meaning 0: Monday, 1: Tuesday, 2: Wednesday, 3: Thursday, 4: Friday, 5: Saturday, 6: Sunday
        plt.xticks(np.arange(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        # set the x axis from 0 to 6
        plt.xlim(0, 6)
        # Set the title
        plt.title('KDE Plot of Purchase Day of the Week')
        # Set x-axis label
        plt.xlabel('Purchase Day of the Week')
        # Set y-axis label
        plt.ylabel('Frequency')
        # point-biserial correlation between purchase_day_of_week and class
        corr_coeff, p_value = pointbiserialr(datetime_columns['purchase_day_of_week'], datetime_columns['class'])
        # put the correlation coefficient and p-value on the plot
        plt.text(0.5, 0.5, 'Correlation Coefficient: %.2f\nP-value: %.2f' % (corr_coeff, p_value),
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    save_plot_as_png(plot_function, 'purchase_kdeplot_day_of_week')


# KDE plot purchase time hour of the day and class
def plot_purchase_datehours_features(data):
    def plot_function():
        datetime_columns = data[['signup_time', 'purchase_time', 'class']].copy()
        pd.set_option('display.max_columns', None)
        # Extract time component (hour) from signup_time
        datetime_columns['purchase_hour'] = datetime_columns['purchase_time'].dt.hour
        # Set up the figure
        plt.figure(figsize=(12, 8))
        # Create KDE plot for the time component (hour)
        sns.kdeplot(data=datetime_columns, x='purchase_hour', hue='class', fill=True, common_norm=False, alpha=0.3)
        # Legend to show class meaning 0: Not Fraud, 1: Fraud
        plt.legend(title='Class', loc='upper right', labels=['Fraud', 'Not Fraud'])
        # Set the x axis from 0 to 23
        plt.xlim(0, 23)
        # set the x axis all the hours of the day
        plt.xticks(np.arange(24))
        # Set the title
        plt.title('KDE Plot of Purchase Hour of the Day')
        # Set x-axis label
        plt.xlabel('Purchase Hour of the Day')
        # Set y-axis label
        plt.ylabel('Frequency')
        # point-biserial correlation between purchase_hour and class
        corr_coeff, p_value = pointbiserialr(datetime_columns['purchase_hour'], datetime_columns['class'])
        # put the correlation coefficient and p-value on the plot
        plt.text(0.5, 0.5, 'Correlation Coefficient: %.2f\nP-value: %.2f' % (corr_coeff, p_value),
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    save_plot_as_png(plot_function, 'purchase_kdeplot_hour_of_day')


# Heatmap with cramer's V of categorical source and class
def cramer_v_categorical_source(data):
    # Function to plot heatmap with crosstab of categorical features and class
    def plot_function():
        # Create a contingency table for the "source" and "class" columns
        contingency_table = pd.crosstab(data['source'], data['class'])
        # Calculate Cramer's V
        cramer_v_value = association(contingency_table.values, method='cramer')
        # Plotting Heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='coolwarm')
        # set custom x-axis labels
        plt.xticks([0.5, 1.5], ['Not Fraud', 'Fraud'], rotation=0)
        # Set the title with Cramer's V value
        plt.title(f'Heatmap of Source and Class\nCramer\'s V: {cramer_v_value:.4f}')
        # Set x-axis label
        plt.xlabel('Class')
        # Set y-axis label
        plt.ylabel('Source')
    save_plot_as_png(plot_function, 'heatmap_categorical_source')


# Heatmap with cramer's V of categorical sex and class column
def cramer_v_categorical_sex(data):
    def plot_function():
        # Create contingency table for sex and class columns
        contingency_table = pd.crosstab(data['sex'], data['class'])
        # Calculate Cramer's V
        cramer_v_value = association(contingency_table.values, method='cramer')
        # Plotting Heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='coolwarm')
        # set custom x-axis labels
        plt.xticks([0.5, 1.5], ['Not Fraud', 'Fraud'], rotation=0)
        # Set the title with Cramer's V value
        plt.title(f'Heatmap of Sex and Class\nCramer\'s V: {cramer_v_value:.4f}')
        # Set x-axis label
        plt.xlabel('Class')
        # Set y-axis label
        plt.ylabel('Sex')
    save_plot_as_png(plot_function, 'heatmap_categorical_sex')
# Heatmap with cramer's V of categorical browser and class column
def cramer_v_categorical_browser(data):
    def plot_function():
        # Create contingency table for browser and class columns
        contingency_table = pd.crosstab(data['browser'], data['class'])
        # Calculate Cramer's V
        cramer_v_value = association(contingency_table.values, method='cramer')
        # Plotting Heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='coolwarm')
        # set custom x-axis labels
        plt.xticks([0.5, 1.5], ['Not Fraud', 'Fraud'], rotation=0)
        # Set the title with Cramer's V value
        plt.title(f'Heatmap of Browser and Class\nCramer\'s V: {cramer_v_value:.4f}')
        # Set x-axis label
        plt.xlabel('Class')
        # Set y-axis label
        plt.ylabel('Browser')
    save_plot_as_png(plot_function, 'heatmap_categorical_browser')

# Heatmap with cramer's V of categorical device_id and class column
# Function to map categorical device_id to numeric values
def map_device_id(data):
    device_id_mapping = {device_id: i for i, device_id in enumerate(data['device_id'].unique())}
    data['device_id_numeric'] = data['device_id'].map(device_id_mapping)
    return data, device_id_mapping

# Function to perform Cramer's V analysis on device_id and class
def cramer_v_categorical_device_id(data):
    def plot_function():
        # Map device_id to numeric values
        data_mapped, device_id_mapping = map_device_id(data)
        # create contingency table for device_id and class columns
        contingency_table = pd.crosstab(data_mapped['device_id'], data_mapped['class'])
        # Calculate Cramer's V
        cramer_v_value = association(contingency_table.values, method='cramer')
        # Create a histogram
        plt.figure(figsize=(12, 8))
        sns.histplot(data_mapped, x='device_id_numeric', hue='class', multiple='stack', bins=30, kde=True,
                     palette='coolwarm')
        # Set the title with Cramer's V value
        plt.title(f'Histogram of Device ID(Mapped) and Class\nCramer\'s V: {cramer_v_value:.4f}')
        # Set x-axis label
        plt.xlabel('Mapped Device ID')
        # Set y-axis label
        plt.ylabel('Count')
        # Show the legend
        plt.legend(title='Class', loc='upper right', labels=['Fraud', 'Not Fraud'])
    # Save plot as PNG using the save_plot_as_png function
    save_plot_as_png(plot_function, 'histogram_categorical_device_id')

#  Horizontal stacked bar chart and cramer's V of categorical country and class column
def cramer_v_categorical_country(data):
    contingency_table = pd.crosstab(data['country'], data['class'])
    cramer_v_value = association(contingency_table.values, method='cramer')

    sorted_countries = contingency_table.sum(axis=1).sort_values(ascending=False)

    top_countries = sorted_countries.head(30)

    other_countries = sorted_countries.tail(len(sorted_countries) - 30).index
    other_sum = contingency_table.loc[other_countries].sum()
    contingency_table = contingency_table.loc[top_countries.index]
    contingency_table.loc['Other'] = other_sum

    # Rename 'class' column to avoid conflicts
    contingency_table = contingency_table.rename(columns={0: 'Not Fraud', 1: 'Fraud'})
    print(contingency_table)
    # invert the order of the rows
    contingency_table = contingency_table.iloc[::-1]

    def plot_function():
        # plot the contingency table as count plot
        contingency_table.plot(kind='barh', stacked=True, figsize=(15, 8), color=['grey', 'coral'])
        # Set the title with Cramer's V value
        plt.title(f'Horizontal Stacked Bar Chart of Country and Class\nCramer\'s V: {cramer_v_value:.4f}')
        # Set x-axis label
        plt.xlabel('Count')
        # Set y-axis label
        plt.ylabel('Country')
        # Show the legend
        plt.legend(title='Class', loc='lower right', labels=['Not Fraud', 'Fraud'])


    # Save plot as PNG using the save_plot_as_png function
    save_plot_as_png(plot_function, 'horizontal_stacked_bar_chart_country')

# Box plot between browser and age
def boxplot_browser_age(data):
    def plot_function():
        # Set up the figure
        plt.figure(figsize=(12, 8))
        # Create boxplot
        sns.boxplot(data=data, x='browser', y='age', hue='class', palette='gray')
        # Set the title
        plt.title('Boxplot of Browser and Age')
        # Set x-axis label
        plt.xlabel('Browser')
        # Set y-axis label
        plt.ylabel('Age')
        # Show the legent with class meaning 0: Not Fraud, 1: Fraud
        plt.legend(title='Class', loc='upper right', labels=['Not Fraud', 'Fraud'])
    save_plot_as_png(plot_function, 'boxplot_browser_age')

# Box plot between country and purchase_value
def boxplot_country_purchase_value(data):
    # Identify the top 15 countries
    top_countries = data['country'].value_counts().head(15).index

    # Create a new column 'country_grouped' to categorize countries
    data['country_grouped'] = np.where(data['country'].isin(top_countries), data['country'], 'Others')

    def plot_function():
        # Set up the figure
        plt.figure(figsize=(18, 10))
        # Create boxplot
        sns.boxplot(data=data, x='country_grouped', y='purchase_value', hue='class', palette='gray')
        # Set the title
        plt.title('Boxplot of Country and Purchase Value')
        # Set x-axis label
        plt.xlabel('Country')
        # rotate the x axis labels
        plt.xticks(rotation=10)
        # Set y-axis label
        plt.ylabel('Purchase Value')
        # Show the legend with class meaning 0: Not Fraud, 1: Fraud
        plt.legend(title='Class', loc='upper right', labels=['Not Fraud', 'Fraud'])

    # Save the plot
    save_plot_as_png(plot_function, 'boxplot_country_purchase_value')

# Box plot between country and age
def boxplot_country_age(data):
    # Identify the top 15 countries
    top_countries = data['country'].value_counts().head(15).index

    # Create a new column 'country_grouped' to categorize countries
    data['country_grouped'] = np.where(data['country'].isin(top_countries), data['country'], 'Others')

    def plot_function():
        # Set up the figure
        plt.figure(figsize=(18, 10))
        # Create boxplot
        sns.boxplot(data=data, x='country_grouped', y='age', hue='class', palette='gray')
        # Set the title
        plt.title('Boxplot of Country and Age')
        # Set x-axis label
        plt.xlabel('Country')
        # rotate the x axis labels
        plt.xticks(rotation=10)
        # Set y-axis label
        plt.ylabel('Age')
        # Show the legend with class meaning 0: Not Fraud, 1: Fraud
        plt.legend(title='Class', loc='upper right', labels=['Not Fraud', 'Fraud'])

    # Save the plot
    save_plot_as_png(plot_function, 'boxplot_country_age')
#

# initialize the python script
if __name__ == '__main__':
    data = load_efraud_dataset('EFraud_Data_Country.csv')
    #data_profiling_report(data)
    #relationship_between_numerical_features_and_label(data)
    relationship_between_datetime_features_and_label(data)
    # datetime_heatmap(data)
    # kde plot sign up time and class
    # plot_sign_up_dateweek_features(data)
    # kde plot hours of the day(signup_time) and class
    # plot_sign_up_datehours_features(data)
    # kde plot purchase date of the week and class
    # plot_purchase_dateweek_features(data)
    # kde plot purchase time hour of the day and class
    # plot_purchase_datehours_features(data)
    # Heatmap with cramer's V of categorical source and class
    #cramer_v_categorical_source(data)
    # Heatmap with cramer' V of categorical sex and class
    #cramer_v_categorical_sex(data)
    # Heatmap with cramer' V of categorical browser and class
    #cramer_v_categorical_browser(data)
    # Heatmap with cramer' V of categorical device_id and class
    #cramer_v_categorical_device_id(data)
    # Horizontal stacked bar chart and cramer's V of categorical country and class
    #cramer_v_categorical_country(data)
    # Box plot between browser and age
    #boxplot_browser_age(data)
    # Box plot between country and purchase_value
    #boxplot_country_purchase_value(data)
    # Box plot between country and age
    #boxplot_country_age(data)