import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.stats import pointbiserialr
from scipy.stats.contingency import association
from statsmodels.graphics.mosaicplot import mosaic
import scipy.stats as stats
import dcor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
            if label.get_text() in [str(int(age_descriptive_stats['min'])), str(int(age_descriptive_stats['25%'])),
                                    str(int(age_descriptive_stats['75%'])), str(int(age_descriptive_stats['max'])),
                                    str(int(age_descriptive_stats['mean']))]:
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
            if label.get_text() in [str(int(purchase_value_descriptive_stats['min'])),
                                    str(int(purchase_value_descriptive_stats['25%'])),
                                    str(int(purchase_value_descriptive_stats['75%'])),
                                    str(int(purchase_value_descriptive_stats['max'])),
                                    str(int(purchase_value_descriptive_stats['mean']))]:
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
        sns.kdeplot(data=data, x='signup_time', hue='class', fill=True, common_norm=True, alpha=0.3,
                    palette=custom_palette)
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
        sns.kdeplot(data=data, x='purchase_time', hue='class', fill=True, common_norm=True, alpha=0.3,
                    palette=custom_palette)
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
        # custom palette
        custom_palette = ['black', 'coral']
        # Create KDE plot for the day of the week component
        sns.kdeplot(data=datetime_columns, x='signup_day_of_week', hue='class', fill=True, common_norm=True, alpha=0.3,
                    palette=custom_palette)
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
        # custom palette
        custom_palette = ['black', 'coral']
        # Create KDE plot for the time component (hour)
        sns.kdeplot(data=datetime_columns, x='signup_hour', hue='class', fill=True, common_norm=True, alpha=0.3,
                    palette=custom_palette)
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
        # custom palette
        custom_palette = ['black', 'coral']
        # Create KDE plot for the day of the week component
        sns.kdeplot(data=datetime_columns, x='purchase_day_of_week', hue='class', fill=True, common_norm=True,
                    alpha=0.3, palette=custom_palette)
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
        # custom palette
        custom_palette = ['black', 'coral']
        # Create KDE plot for the time component (hour)
        sns.kdeplot(data=datetime_columns, x='purchase_hour', hue='class', fill=True, common_norm=True, alpha=0.3,
                    palette=custom_palette)
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


# Heatmap with relative frequncy and total, cramer's V of categorical source and class
def cramer_v_categorical_source(data):
    def plot_function():
        # create contingency table for source and class columns with relative frequency and total
        contingency_table = pd.crosstab(data['source'], data['class'], normalize=True, margins=True)
        # custom cmap
        custom_palette = sns.diverging_palette(220, 20, as_cmap=True)
        # Plot the heatmap
        sns.heatmap(contingency_table, annot=True, fmt=".2f", cmap=custom_palette)
        # Set x-axis label
        plt.xlabel('Class')
        # Set y-axis label
        plt.ylabel('Source')
        # Set custom x-axis labels
        plt.xticks([0.5, 1.5, 2.5], ['Not Fraud', 'Fraud', 'All'], rotation=0)
        # Calculate Cramer's V  value
        cross_tab = pd.crosstab(data['source'], data['class'])
        cramer_v_value = association(cross_tab.values, method='cramer')
        # set the title with Cramer's V value
        plt.title(f'Heatmap of Source and Class\nCramer\'s V: {cramer_v_value:.4f}')

    # Save the plot as PNG
    save_plot_as_png(plot_function, 'heatmap_categorical_source')


# Heatmap with relative frequncy and total, cramer's V of categorical sex and class column
def cramer_v_categorical_sex(data):
    def plot_function():
        # create contingency table for sex and class columns with relative frequency and total
        contingency_table = pd.crosstab(data['sex'], data['class'], normalize=True, margins=True)
        # custom cmap
        custom_palette = sns.diverging_palette(220, 20, as_cmap=True)
        # Plot the heatmap
        sns.heatmap(contingency_table, annot=True, fmt=".2f", cmap=custom_palette)
        # Set x-axis label
        plt.xlabel('Class')
        # Set y-axis label
        plt.ylabel('Sex')
        # Set custom x-axis labels
        plt.xticks([0.5, 1.5, 2.5], ['Not Fraud', 'Fraud', 'All'], rotation=0)
        # Calculate Cramer's V  value
        cross_tab = pd.crosstab(data['sex'], data['class'])
        cramer_v_value = association(cross_tab.values, method='cramer')
        # set the title with Cramer's V value
        plt.title(f'Heatmap of Sex and Class\nCramer\'s V: {cramer_v_value:.4f}')

    # Save the plot as PNG
    save_plot_as_png(plot_function, 'heatmap_categorical_sex')


# Heatmap with relative frequncy and total, cramer's V of categorical browser and class column
def cramer_v_categorical_browser(data):
    def plot_function():
        # create contingency table for browser and class columns with relative frequency and total
        contingency_table = pd.crosstab(data['browser'], data['class'], normalize=True, margins=True)
        # custom cmap
        custom_palette = sns.diverging_palette(220, 20, as_cmap=True)
        # Plot the heatmap
        sns.heatmap(contingency_table, annot=True, fmt=".3f", cmap=custom_palette)
        # Set x-axis label
        plt.xlabel('Class')
        # Set y-axis label
        plt.ylabel('Browser')
        # Set custom x-axis labels
        plt.xticks([0.5, 1.5, 2.5], ['Not Fraud', 'Fraud', 'All'], rotation=0)
        # Calculate Cramer's V  value
        cross_tab = pd.crosstab(data['browser'], data['class'])
        cramer_v_value = association(cross_tab.values, method='cramer')
        # set the title with Cramer's V value
        plt.title(f'Heatmap of Browser and Class\nCramer\'s V: {cramer_v_value:.4f}')

    # Save the plot as PNG
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
        data_mapped, device_id_mapping = map_device_id(data.copy())

        contingency_table = pd.crosstab(data_mapped['device_id'], data_mapped['class'])
        # Calculate Cramer's V
        cramer_v_value = association(contingency_table.values, method='cramer')
        plt.figure(figsize=(12, 8))
        custom_palette = ['grey', 'coral']
        sns.histplot(data=data_mapped, x='device_id_numeric', hue='class', stat='probability', bins=30, kde=True,
                     multiple='stack', palette=custom_palette)
        # Set the title with Cramer's V value
        plt.title(f'Histogram of Device ID(Mapped) and Class\nCramer\'s V: {cramer_v_value:.4f}')
        plt.xlabel('Mapped Device ID to numeric values')
        plt.ylabel('Relative Frequency')
        plt.legend(title='Class', loc='upper right', labels=['Fraud', 'Not Fraud'])

    save_plot_as_png(plot_function, 'histogram_categorical_device_id')


#  Horizontal stacked bar chart and cramer's V of categorical country and class column
def cramer_v_categorical_country(data):
    # Calculate Cramer's V
    contingency_cramer_v = pd.crosstab(data['country'], data['class'])
    cramer_v_value = association(contingency_cramer_v.values, method='cramer')
    # create contingency table for country and class columns relative frequency
    contingency_table = pd.crosstab(data['country'], data['class'], normalize=True, margins=True)
    contingency_table = contingency_table.sort_values(by='All', ascending=False)
    contingency_table = contingency_table.drop('All', axis=1)
    contingency_table = contingency_table.drop('All', axis=0)
    # take the top 30 countries
    top_countries = contingency_table.head(30).copy()
    # calculate the sum of relative frequencies for other countries
    other_countries_frequency = contingency_table[30:].sum()
    top_countries.loc['Others'] = other_countries_frequency
    top_countries = top_countries.rename(columns={'0': 'Not Fraud', '1': 'Fraud'})
    top_countries = top_countries.iloc[::-1]

    def plot_function():
        # Create a horizontal stacked bar chart
        top_countries.plot(kind='barh', stacked=True, figsize=(15, 8), color=['grey', 'coral'])
        # Set the title with Cramer's V value
        plt.title(f'Horizontal Stacked Bar Chart of Country and Class\nCramer\'s V: {cramer_v_value:.4f}')
        # Set x-axis label
        plt.xlabel('Relative Frequency')
        # Set y-axis label
        plt.ylabel('Country')
        # Show the legend with class meaning 0: Not Fraud, 1: Fraud
        plt.legend(title='Class', loc='lower right', labels=['Not Fraud', 'Fraud'])

    # Save the plot as PNG
    save_plot_as_png(plot_function, 'bar_chart_categorical_country')


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


# Line plot between age and purchase_value with distance correlation segmentation analysis
def scatter_plot_age_purchase_value(data):
    def plot_function():
        # Calculate the average purchase value for each age group and class
        average_purchase_by_age_group_class = data.groupby(['age', 'class'], observed=True)[
            'purchase_value'].mean().unstack()
        print(average_purchase_by_age_group_class)
        # Set up the figure
        plt.figure(figsize=(12, 8))
        # custom palette
        custom_palette = ['grey', 'coral']
        # Create line plot
        # Create line plot
        sns.lineplot(data=average_purchase_by_age_group_class, marker='o', palette=custom_palette, dashes=False,
                     markersize=10, linewidth=2)

        # Set x-axis label
        plt.xlabel('Age')
        # Set y-axis label
        # set x axis every 2 years
        plt.xticks(np.arange(data['age'].min(), data['age'].max(), 2))
        # set y axis every 5
        plt.yticks(np.arange(0, 70, 5))
        plt.ylabel('Average Purchase Value')
        # Show the legend with class meaning 0: Not Fraud, 1: Fraud using dictionary
        plt.legend(title='Class', handles=[
            plt.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=10, label='Not Fraud'),
            plt.Line2D([], [], color='coral', marker='o', linestyle='None', markersize=10, label='Fraud')
        ], loc='upper right')
        fraud_data = data[data['class'] == 1]
        distance_correlation = dcor.distance_correlation(fraud_data['age'].astype(float),
                                                         fraud_data['purchase_value'].astype(float))
        not_fraud_data = data[data['class'] == 0]
        distance_correlation_not_fraud = dcor.distance_correlation(not_fraud_data['age'].astype(float),
                                                                   not_fraud_data['purchase_value'].astype(float))
        plt.title(
            f'Line Plot of Age and Average Purchase Value\nDistance Correlation(Fraud): {distance_correlation:.4f}\nDistance Correlation(Not Fraud): {distance_correlation_not_fraud:.4f}')

    # Save the plot
    save_plot_as_png(plot_function, 'scatterplot_age_purchase_value')


# count plot between user_id and device_id
def number_user_id_per_device_id(data):
    def plot_function():
        fraud_data = data[data['class'] == 1]
        fraud_user_device_count = fraud_data.groupby('device_id')['user_id'].nunique().reset_index(
            name='fraud_user_per_device')
        not_fraud_data = data[data['class'] == 0]
        not_fraud_user_device_count = not_fraud_data.groupby('device_id')['user_id'].nunique().reset_index(
            name='not_fraud_user_per_device')
        user_device_count = fraud_user_device_count.merge(not_fraud_user_device_count, how='outer', on='device_id')

        # Scatter plot of device_id and
        plt.figure(figsize=(12, 8))
        plt.scatter(user_device_count['device_id'], user_device_count['not_fraud_user_per_device'], label='Not Fraud',
                    color='grey', alpha=0.3, marker='o', s=100)
        plt.scatter(user_device_count['device_id'], user_device_count['fraud_user_per_device'], label='Fraud',
                    color='coral', marker='x', s=20, alpha=0.3)
        # Set the title
        plt.title('Scatter Plot Number of Users per Device ID')
        # Set x-axis label
        plt.xlabel('Device ID')
        # remove labels from x-axis
        plt.xticks([])
        # Set y-axis label
        plt.ylabel('Number of Users')
        # y axis 1 by 1
        plt.yticks(np.arange(0, 20, 1))
        # Show the legend with class meaning 0: Not Fraud, 1: Fraud
        plt.legend(title='Class', loc='upper right', labels=['Not Fraud', 'Fraud'])

    save_plot_as_png(plot_function, 'lineplot_user_id_device_id')


# relationship between source and browser
def source_browser_relationship(data):
    def plot_function():
        fraud_data = data[data['class'] == 1]
        contingency_table = pd.crosstab(fraud_data['source'], fraud_data['browser'], margins=True, normalize=True)
        custom_cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(contingency_table, annot=True, fmt=".2f", cmap=custom_cmap)
        plt.xlabel('Browser')
        plt.ylabel('Source')
        # Calculate Cramer's V  value
        cross_tab = pd.crosstab(fraud_data['source'], fraud_data['browser'])
        cramer_v_value = association(cross_tab.values, method='cramer')
        # set the title with Cramer's V value
        plt.title(f'Heatmap of Source and Browser\nCramer\'s V: {cramer_v_value:.4f}')

    save_plot_as_png(plot_function, 'heatmap_source_browser')


# Relationship between source and country
def source_country_relationship(data):
    def plot_function():
        fraud_data = data[data['class'] == 1]
        contingency_table = pd.crosstab(fraud_data['country'], fraud_data['source'], margins=True, normalize=True)
        # sort the contingency table by the total column
        contingency_table = contingency_table.sort_values(by='All', ascending=False)
        contingency_table = contingency_table.drop('All', axis=1)
        contingency_table = contingency_table.drop('All', axis=0)
        # take the top 30 countries
        top_countries = contingency_table.head(30).copy()
        # calculate the sum of relative frequencies for other countries
        other_countries_frequency = contingency_table[30:].sum()
        top_countries.loc['Others'] = other_countries_frequency

        top_countries = top_countries.iloc[::-1]

        top_countries.plot(kind='barh', stacked=True, figsize=(15, 8))
        # Cramer's V  value
        contingency_table_cramer_v = pd.crosstab(fraud_data['country'], fraud_data['source'])
        cramer_v_value = association(contingency_table_cramer_v.values, method='cramer')
        # set the title with Cramer's V value
        plt.title(f'Horizontal Stacked Bar Chart of Country and Source\nCramer\'s V: {cramer_v_value:.4f}')
        plt.xlabel('Fraud Relative Frequency')

    save_plot_as_png(plot_function, 'bar_chart_country_source')


# relationship between browser and device_id
def browser_device_id_relationship(data):
    def plot_function():
        ddata = data[['device_id', 'browser', 'class']].copy()
        # find which browser are in fraud and not are in not fraud
        fraud_data = ddata[ddata['class'] == 1]
        not_fraud_data = ddata[ddata['class'] == 0]
        # merge the fraud with not fraud data on device id and browser leaving only device id and browser that are fraud
        fraud_browser_per_device = fraud_data.merge(not_fraud_data, how='left', on=['device_id'],
                                                    indicator=True)
        different_browsers = fraud_browser_per_device[
            fraud_browser_per_device['browser_x'] != fraud_browser_per_device['browser_y']]

        plt.figure(figsize=(12, 8))
        # Create a horizontal stacked bar chart
        sns.countplot(data=different_browsers, x='_merge', hue='browser_x', palette='gray')
        plt.xticks([0, 2], ['Fraud Cases', 'Different Browsers for Fraud and Legitimate'])

        # Cramer V  value
        fraud_ddata = ddata[ddata['class'] == 1]
        contingency_table_cramer_v = pd.crosstab(fraud_ddata['browser'], fraud_data['device_id'])
        cramer_v_value = association(contingency_table_cramer_v.values, method='cramer')

        plt.ylabel('Number of Devices')
        plt.xlabel("Browser in Fraud and Legitimate transactions")
        plt.title(f'Count Plot of Browser and Device ID\nCramer\'s V: {cramer_v_value:.2f}')

    save_plot_as_png(plot_function, 'heatmap_browser_device_id')
# relationship between country and browser
def country_browser_relationship(data):
    def plot_function():
        fraud_data = data[data['class'] == 1]
        contingency_table = pd.crosstab(fraud_data['country'], fraud_data['browser'], margins=True, normalize=True)
        # sort the contingency table by the total column
        contingency_table = contingency_table.sort_values(by='All', ascending=False)
        contingency_table = contingency_table.drop('All', axis=1)
        contingency_table = contingency_table.drop('All', axis=0)
        # take the top 30 countries
        top_countries = contingency_table.head(30).copy()
        # calculate the sum of relative frequencies for other countries
        other_countries_frequency = contingency_table[30:].sum()
        top_countries.loc['Others'] = other_countries_frequency
        top_countries = top_countries.iloc[::-1]
        top_countries.plot(kind='barh', stacked=True, figsize=(15, 8))
        # Cramer's V  value
        contingency_table_cramer_v = pd.crosstab(fraud_data['country'], fraud_data['browser'])
        cramer_v_value = association(contingency_table_cramer_v.values, method='cramer')
        # set the title with Cramer's V value
        plt.title(f'Horizontal Stacked Bar Chart of Country and Browser\nCramer\'s V: {cramer_v_value:.4f}')
        plt.xlabel('Fraud Relative Frequency')

    save_plot_as_png(plot_function, 'heatmap_country_browser')

# initialize the python script
if __name__ == '__main__':
    data = load_efraud_dataset('EFraud_Data_Country.csv')
    # data_profiling_report(data)
    # relationship_between_numerical_features_and_label(data)
    # relationship_between_datetime_features_and_label(data)

    # kde plot sign up time and class
    # plot_sign_up_dateweek_features(data)
    # kde plot hours of the day(signup_time) and class
    # plot_sign_up_datehours_features(data)
    # kde plot purchase date of the week and class
    # plot_purchase_dateweek_features(data)
    # kde plot purchase time hour of the day and class
    # plot_purchase_datehours_features(data)
    # Heatmap with cramer's V of categorical source and class
    # cramer_v_categorical_source(data)
    # Heatmap with cramer' V of categorical sex and class
    # cramer_v_categorical_sex(data)
    # Heatmap with cramer' V of categorical browser and class
    # cramer_v_categorical_browser(data)
    # Heatmap with cramer' V of categorical device_id and class
    # cramer_v_categorical_device_id(data)
    # Horizontal stacked bar chart and cramer's V of categorical country and class
    # cramer_v_categorical_country(data)
    # Box plot between browser and age
    # boxplot_browser_age(data)
    # Box plot between country and purchase_value
    # boxplot_country_purchase_value(data)
    # Box plot between country and age
    # boxplot_country_age(data)
    # Heatmap between age and purchase_value
    # scatter_plot_age_purchase_value(data)
    ############ Categorical-Categorical Analysis ################
    # number_user_id_per_device_id(data)
    # source_browser_relationship(data)
    # source_country_relationship(data)
    #browser_device_id_relationship(data)
    country_browser_relationship(data)
