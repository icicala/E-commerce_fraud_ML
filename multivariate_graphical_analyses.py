import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from kfda import Kfda, kfda
from scipy.cluster.hierarchy import linkage, dendrogram

from scipy.linalg import eigh
from scipy.spatial.distance import pdist
from scipy.stats import pointbiserialr
from scipy.stats.contingency import association
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from statsmodels.graphics.mosaicplot import mosaic
import scipy.stats as stats
import dcor
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.preprocessing import StandardScaler

from ydata_profiling import ProfileReport
from pandas.plotting import autocorrelation_plot
from scipy import fft
from scipy.stats import spearmanr

from SBS import SBS


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
        custom_palette = ['grey', 'coral']
        ax = sns.countplot(data=data, x='age', hue='class', stat='proportion', palette=custom_palette)
        plt.title('Correlation between Age and Class')
        plt.xlabel('Age (in years')
        plt.ylabel('Relative Frequency')
        plt.legend(title='Class', loc='upper right', labels=['Not Fraud', 'Fraud'])
        for ind, label in enumerate(ax.get_xticklabels()):
            age_descriptive_stats = data['age'].describe()
            ax.xticks = []
            if label.get_text() in [str(int(age_descriptive_stats['min'])), str(int(age_descriptive_stats['25%'])),
                                    str(int(age_descriptive_stats['75%'])), str(int(age_descriptive_stats['max'])),
                                    str(int(age_descriptive_stats['mean']))]:
                label.set_visible(True)
            else:
                label.set_visible(False)
        corr_coeff, p_value = pointbiserialr(data['age'], data['class'])
        plt.text(0.5, 0.5, 'Correlation Coefficient: %.2f\nP-value: %.2f' % (corr_coeff, p_value),
                 horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    save_plot_as_png(plot_age, 'countplot_age')

    def plot_purchase_value():

        custom_palette = ['grey', 'coral']
        ax = sns.countplot(data=data, x='purchase_value', hue='class', stat='proportion', palette=custom_palette)

        plt.title('Correlation between Purchase Value and Class')
        plt.xlabel('Purchase Value (in USD)')
        plt.ylabel('Relative Frequency')
        plt.legend(title='Class', loc='upper right', labels=['Not Fraud', 'Fraud'])
        for ind, label in enumerate(ax.get_xticklabels()):
            purchase_value_descriptive_stats = data['purchase_value'].describe()
            ax.xticks = []
            if label.get_text() in [str(int(purchase_value_descriptive_stats['min'])),
                                    str(int(purchase_value_descriptive_stats['25%'])),
                                    str(int(purchase_value_descriptive_stats['75%'])),
                                    str(int(purchase_value_descriptive_stats['max'])),
                                    str(int(purchase_value_descriptive_stats['mean']))]:
                label.set_visible(True)
            else:
                label.set_visible(False)
        corr_coeff, p_value = pointbiserialr(data['purchase_value'], data['class'])
        plt.text(0.5, 0.5, 'Correlation Coefficient: %.2f\nP-value: %.2f' % (corr_coeff, p_value),
                 horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    save_plot_as_png(plot_purchase_value, 'countplot_purchase_value')


# count plot of correlation between datetime features(signup_time, purchase_time) and label(class) column
def relationship_between_datetime_features_and_label(data):
    def plot_signup_time():
        data['month'] = data['signup_time'].dt.month
        monthly_data = data.groupby(['month', 'class']).size().unstack(fill_value=0)
        monthly_data_relative = monthly_data.divide(monthly_data.sum(axis=0).sum(), axis=0)
        plt.figure(figsize=(14, 7))
        custom_palette = ['grey', 'coral']
        sns.lineplot(data=monthly_data_relative, marker='o', palette=custom_palette, dashes=False, markersize=10, linewidth=2)
        plt.title('Relationship between Signup Time and Class per Month')
        plt.xlabel('Month')
        plt.ylabel('Relative Frequency')
        plt.xticks(ticks=range(1, 9), labels=[
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug'])

        plt.legend(title='Class', loc='upper right', handles=[
            plt.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=10, label='Not Fraud'),
            plt.Line2D([], [], color='coral', marker='o', linestyle='None', markersize=10, label='Fraud')
        ])

        plt.grid(True)

    save_plot_as_png(plot_signup_time, 'lineplot_signup_time')

    def plot_purchase_time():
        data['month'] = data['purchase_time'].dt.month
        monthly_data = data.groupby(['month', 'class']).size().unstack(fill_value=0)
        monthly_data_relative = monthly_data.divide(monthly_data.sum(axis=0).sum(), axis=0)
        plt.figure(figsize=(14, 7))
        custom_palette = ['grey', 'coral']
        sns.lineplot(data=monthly_data_relative, marker='o', palette=custom_palette, dashes=False, markersize=10, linewidth=2)
        plt.title('Relationship between Purchase Time and Class per Month')
        plt.xlabel('Month')
        plt.ylabel('Relative Frequency')
        plt.xticks(ticks=range(1, 13), labels=[
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        plt.legend(title='Class', loc='upper right', handles=[
            plt.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=10, label='Not Fraud'),
            plt.Line2D([], [], color='coral', marker='o', linestyle='None', markersize=10, label='Fraud')
        ])

        plt.grid(True)

    save_plot_as_png(plot_purchase_time, 'lineplot_purchase_time')

# lineplot of signup time weekly and class
def plot_sign_up_weekly(data):
    def plot_function():
        # Create a new column 'signup_day_of_week' to categorize the day of the week
        data['signup_week'] = data['signup_time'].dt.isocalendar().week
        weekly_data = data.groupby(['signup_week', 'class']).size().unstack(fill_value=0)
        weekly_data_relative = weekly_data.divide(weekly_data.sum(axis=0).sum(), axis=0)
        plt.figure(figsize=(14, 7))
        custom_palette = ['grey', 'coral']
        sns.lineplot(data=weekly_data_relative, marker='o', palette=custom_palette, dashes=False, markersize=10, linewidth=2)
        plt.title('Relationship between Signup Time and Class per Week')
        plt.xlabel('Week')
        # x-axis from 1 to 52
        plt.xticks(ticks=range(1, 35))
        plt.ylabel('Relative Frequency')
        plt.legend(title='Class', loc='upper right', handles=[
            plt.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=10, label='Not Fraud'),
            plt.Line2D([], [], color='coral', marker='o', linestyle='None', markersize=10, label='Fraud')
        ])
        plt.grid(True)

    save_plot_as_png(plot_function, 'lineplot_signup_week')

# Line plot of signup time day of the month and class
def plot_sign_up_daymonth_features(data):
    def plot_signup_day():
        # day of the month
        data['signup_day'] = data['signup_time'].dt.day
        daily_data = data.groupby(['signup_day', 'class']).size().unstack(fill_value=0)
        daily_data_relative = daily_data.divide(daily_data.sum(axis=0).sum(), axis=0)
        plt.figure(figsize=(14, 7))
        custom_palette = ['grey', 'coral']
        sns.lineplot(data=daily_data_relative, marker='o', palette=custom_palette, dashes=False, markersize=10, linewidth=2)
        plt.title('Relationship between Signup Time and Class per Day of the Month')
        plt.xlabel('Day of the Month')
        plt.ylabel('Relative Frequency')
        # x-axis from 1 to 31
        plt.xticks(ticks=range(1, 32))
        plt.legend(title='Class', loc='upper right', handles=[
            plt.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=10, label='Not Fraud'),
            plt.Line2D([], [], color='coral', marker='o', linestyle='None', markersize=10, label='Fraud')
        ])
        plt.grid(True)

    save_plot_as_png(plot_signup_day, 'lineplot_signup_day')

# Line plot of signup day of the week and class
def plot_sign_up_dateweek_features(data):
    def plot_signup_day():
        # filter fraud transactions
        fraud_data = data[data['class'] == 1].copy()
        fraud_data['signup_day_of_week'] = fraud_data['signup_time'].dt.dayofweek
        daily_data = fraud_data.groupby(['signup_day_of_week', 'class']).size().unstack(fill_value=0)
        daily_data_relative = daily_data.divide(daily_data.sum(axis=0).sum(), axis=0)
        plt.figure(figsize=(14, 7))
        custom_palette = ['coral']
        sns.lineplot(data=daily_data_relative, marker='o', palette=custom_palette, dashes=False, markersize=10, linewidth=2)
        plt.title('Relationship between Signup Time and Fraud transactions per Day')
        plt.xlabel('Day of the Week')
        plt.ylabel('Fraud Relative Frequency')
        # remove legend
        plt.legend().remove()
        plt.xticks(ticks=range(0, 7), labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        plt.grid(True)

    save_plot_as_png(plot_signup_day, 'lineplot_signup_day_of_week')
# line plot of signup time hour of the day and class
def plot_sign_up_datehours_features(data):
    def plot_signup_hour():
        # filter fraud transactions
        fraud_data = data[data['class'] == 1].copy()
        fraud_data['signup_hour'] = fraud_data['signup_time'].dt.hour
        hourly_data = fraud_data.groupby(['signup_hour', 'class']).size().unstack(fill_value=0)
        hourly_data_relative = hourly_data.divide(hourly_data.sum(axis=0).sum(), axis=0)
        plt.figure(figsize=(14, 7))
        custom_palette = ['coral']
        sns.lineplot(data=hourly_data_relative, marker='o', palette=custom_palette, dashes=False, markersize=10, linewidth=2)
        plt.title('Relationship between Signup Time and Fraud transactions per Hour')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Fraud Relative Frequency')
        # remove legend
        plt.legend().remove()
        plt.xticks(ticks=range(0, 24))
        plt.grid(True)

    save_plot_as_png(plot_signup_hour, 'lineplot_signup_hour')

# kde plot hours of the day(signup_time) and class



# line plot of purchase time week of the year and class
def plot_purchase_weekly(data):
    def function_plot():
        # Create a new column 'purchase_week' to categorize the week of the year
        data['purchase_week'] = data['purchase_time'].dt.isocalendar().week
        weekly_data = data.groupby(['purchase_week', 'class']).size().unstack(fill_value=0)
        weekly_data_relative = weekly_data.divide(weekly_data.sum(axis=0).sum(), axis=0)
        plt.figure(figsize=(14, 7))
        custom_palette = ['grey', 'coral']
        sns.lineplot(data=weekly_data_relative, marker='o', palette=custom_palette, dashes=False, markersize=10, linewidth=2)
        plt.title('Relationship between Purchase Time and Class per Week')
        plt.xlabel('Week')
        plt.ylabel('Relative Frequency')
        plt.xticks(ticks=range(1, 53))
        plt.legend(title='Class', loc='upper right', handles=[
            plt.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=10, label='Not Fraud'),
            plt.Line2D([], [], color='coral', marker='o', linestyle='None', markersize=10, label='Fraud')
        ])
        plt.grid(True)

    save_plot_as_png(function_plot, 'lineplot_purchase_week')

# line plot of purchase time day of the month and class
def plot_purchase_daymonth_features(data):
    def plot_purchase_day():
        # day of the month
        data['purchase_day'] = data['purchase_time'].dt.day
        daily_data = data.groupby(['purchase_day', 'class']).size().unstack(fill_value=0)
        daily_data_relative = daily_data.divide(daily_data.sum(axis=0).sum(), axis=0)
        plt.figure(figsize=(14, 7))
        custom_palette = ['grey', 'coral']
        sns.lineplot(data=daily_data_relative, marker='o', palette=custom_palette, dashes=False, markersize=10, linewidth=2)
        plt.title('Relationship between Purchase Time and Class per Day of the Month')
        plt.xlabel('Day of the Month')
        plt.ylabel('Relative Frequency')
        plt.xticks(ticks=range(1, 32))
        plt.legend(title='Class', loc='upper right', handles=[
            plt.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=10, label='Not Fraud'),
            plt.Line2D([], [], color='coral', marker='o', linestyle='None', markersize=10, label='Fraud')
        ])
        plt.grid(True)

    save_plot_as_png(plot_purchase_day, 'lineplot_purchase_day')


# line plot of purchase time day of the week and class
def plot_purchase_dateweek_features(data):
    def plot_purchase_day():
        # filter fraud transactions

        data['purchase_day_of_week'] = data['purchase_time'].dt.dayofweek
        daily_data = data.groupby(['purchase_day_of_week', 'class']).size().unstack(fill_value=0)
        daily_data_relative = daily_data.divide(daily_data.sum(axis=0).sum(), axis=0)
        plt.figure(figsize=(14, 7))
        custom_palette = ['grey', 'coral']
        sns.lineplot(data=daily_data_relative, marker='o', palette=custom_palette, dashes=False, markersize=10, linewidth=2)
        plt.title('Relationship between Purchase Time and Fraud transactions per Day')
        plt.xlabel('Day of the Week')
        plt.ylabel('Fraud Relative Frequency')
        plt.legend(title='Class', loc='upper right', handles=[
            plt.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=10, label='Not Fraud'),
            plt.Line2D([], [], color='coral', marker='o', linestyle='None', markersize=10, label='Fraud')
        ])
        plt.xticks(ticks=range(0, 7), labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        plt.grid(True)

    save_plot_as_png(plot_purchase_day, 'lineplot_purchase_day_of_week')

# line plot of purchase time hour of the day and class
def plot_purchase_datehours_features(data):
    def plot_purchase_hour():
        # filter fraud transactions
        fraud_data = data[data['class'] == 1].copy()
        fraud_data['purchase_hour'] = fraud_data['purchase_time'].dt.hour
        hourly_data = fraud_data.groupby(['purchase_hour', 'class']).size().unstack(fill_value=0)
        hourly_data_relative = hourly_data.divide(hourly_data.sum(axis=0).sum(), axis=0)
        plt.figure(figsize=(14, 7))
        custom_palette = ['coral']
        sns.lineplot(data=hourly_data_relative, marker='o', palette=custom_palette, dashes=False, markersize=10, linewidth=2)
        plt.title('Relationship between Purchase Time and Fraud transactions per Hour')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Fraud Relative Frequency')
        plt.xticks(ticks=range(0, 24))
        plt.grid(True)
        plt.legend().remove()

    save_plot_as_png(plot_purchase_hour, 'lineplot_purchase_hour')
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

# relationship between signup time and purchase time
def signup_purchase_time_relationship(data):
    def plot_function():

        data['time_difference'] = (data['purchase_time'] - data['signup_time'])
        data['time_difference_sec'] = data['time_difference'].dt.total_seconds()
        plt.figure(figsize=(12, 8))
        custom_palette = ['grey', 'coral']
        sns.boxenplot(data=data, x='class', y='time_difference_sec', hue='class', palette=custom_palette)
        plt.title('Time Differences for Fraud and Legitimate Transactions')
        plt.xlabel('Class')
        plt.xticks([0, 1], ['Not Fraud', 'Fraud'])
        plt.ylabel('Time Difference in Seconds')
        plt.legend().remove()

    save_plot_as_png(plot_function, 'boxplot_difference_time_seconds')
# Purchase value and Purchase time relationship per hour
def purchase_value_purchase_time_relationship(data):
    def plot_function():
        data['purchase_hour'] = data['purchase_time'].dt.hour
        plt.figure(figsize=(12, 8))
        custom_palette = ['grey', 'coral']
        plt.title('Purchase Value and Purchase Time per Hour')
        plt.xlabel('Purchase Hour')
        plt.ylabel('Purchase Value')
        plt.xticks(np.arange(24))
        sns.lineplot(data=data, x='purchase_hour', y='purchase_value', hue='class', palette=custom_palette,
                     linestyle='--')
        plt.legend(title='Class', handles=[
            plt.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=10, label='Not Fraud'),
            plt.Line2D([], [], color='coral', marker='o', linestyle='None', markersize=10, label='Fraud')
        ], loc='upper right')

    save_plot_as_png(plot_function, 'scatterplot_purchase_value_purchase_time_hour')
# relationship between signup time and device_id
def signup_device_id_relationship(data):
    def plot_function():
        signup_counts = data.groupby(['device_id', 'class'])['signup_time'].count().reset_index()
        custom_palette = ['grey', 'coral']
        plt.figure(figsize=(12, 8))
        sns.stripplot(data=signup_counts, x='class', y='signup_time', hue='class', palette= custom_palette, jitter=0.3, legend=False)
        plt.title('Number of Sign Ups and Device ID')
        plt.xlabel('Class')
        plt.ylabel('Number of Sign Ups')
        sns.despine()
        plt.yticks(np.arange(0, 20, 1))
        plt.xticks([0, 1], ['Not Fraud', 'Fraud'])

    save_plot_as_png(plot_function, 'stripplot_device_id_class')
# Linear Discrimant Analysis
def lda_analysis(data):

    # extract from purchase value year, month, day, hour, minute, second
    data['purchase_year'] = data['purchase_time'].dt.year
    data['purchase_month'] = data['purchase_time'].dt.month
    data['purchase_day'] = data['purchase_time'].dt.day
    data['purchase_hour'] = data['purchase_time'].dt.hour
    data['purchase_minute'] = data['purchase_time'].dt.minute
    data['purchase_second'] = data['purchase_time'].dt.second
    # extract from signup value year, month, day, hour, minute, second
    data['signup_year'] = data['signup_time'].dt.year
    data['signup_month'] = data['signup_time'].dt.month
    data['signup_day'] = data['signup_time'].dt.day
    data['signup_hour'] = data['signup_time'].dt.hour
    data['signup_minute'] = data['signup_time'].dt.minute
    data['signup_second'] = data['signup_time'].dt.second

    # count and map device_id
    device_id_map = data['device_id'].value_counts().to_dict()
    data['device_id_count'] = data['device_id'].map(device_id_map)
    # count and map source
    source_map = data['source'].value_counts().to_dict()
    data['source_count'] = data['source'].map(source_map)
    # count and map browser
    browser_map = data['browser'].value_counts().to_dict()
    data['browser_count'] = data['browser'].map(browser_map)
    # count and map sex
    sex_map = data['sex'].value_counts().to_dict()
    data['sex_count'] = data['sex'].map(sex_map)
    # count and map country
    country_map = data['country'].value_counts().to_dict()
    data['country_count'] = data['country'].map(country_map)
    # count and map ip_address
    ip_address_map = data['ip_address'].value_counts().to_dict()
    data['ip_address_count'] = data['ip_address'].map(ip_address_map)

    # drop unnecessary columns
    data = data.drop(['signup_time', 'purchase_time', 'device_id', 'user_id', 'source', 'browser', 'sex', 'country', 'purchase_year', 'signup_year', 'ip_address'], axis=1)
    # split the data into features and target
    data_features = data.drop('class', axis=1)
    data_label = data['class']
    # Standardize the data
    scaler = StandardScaler()
    data_scale = scaler.fit_transform(data_features)
    mean_vectors = []
    # for each class calculate the mean vector
    for cl in range(0, 2):
        mean_vectors.append(np.mean(data_scale[data_label == cl], axis=0))

    # calculate the within-class scatter matrix for unbalanced data
    within_class_scatter_matrix = np.zeros((data_scale.shape[1], data_scale.shape[1]))
    for cl, mv in zip(range(0, 2), mean_vectors):
        class_scatter = np.cov(data_scale[data_label == cl].T)
        within_class_scatter_matrix += class_scatter
    # calculate the between-class scatter matrix
    overall_mean = np.mean(data_scale, axis=0)
    between_class_scatter_matrix = np.zeros((data_scale.shape[1], data_scale.shape[1]))
    for i, mean_vec in enumerate(mean_vectors):
        n = data_scale[data_label == i + 1, :].shape[0]
        mean_vec = mean_vec.reshape(data_scale.shape[1], 1)
        overall_mean = overall_mean.reshape(data_scale.shape[1], 1)
        between_class_scatter_matrix += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
        # print between class scatter matrix
    # calculate the eigenvalues and eigenvectors
    eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(within_class_scatter_matrix).dot(between_class_scatter_matrix))
    # sort the eigenvalues and eigenvectors
    eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

    # get the LDA coefficients
    lda_coefficients = eigen_pairs[0][1].real
    features_names = data_features.columns
    importance = pd.Series(lda_coefficients, index=features_names)
    sorted_importance = importance.abs().sort_values(ascending=False)
    sorted_importance = sorted_importance[::-1]
    def plot_function():
        plt.figure(figsize=(15, 10))
        sorted_importance.plot(kind='barh', color='coral')
        # log scale
        plt.xscale('log')
        plt.title('LDA Coefficients for Features relationship with Class')
        plt.ylabel('Feature')
        plt.xlabel('Coefficient values')

    save_plot_as_png(plot_function, 'lda_coefficients')
# Kernel Discriminant Analysis
def sbs_analysis(data):
    # extract from purchase value year, month, day, hour, minute, second
    data['purchase_year'] = data['purchase_time'].dt.year
    data['purchase_month'] = data['purchase_time'].dt.month
    data['purchase_day'] = data['purchase_time'].dt.day
    data['purchase_hour'] = data['purchase_time'].dt.hour
    data['purchase_minute'] = data['purchase_time'].dt.minute
    data['purchase_second'] = data['purchase_time'].dt.second
    # extract from signup value year, month, day, hour, minute, second
    data['signup_year'] = data['signup_time'].dt.year
    data['signup_month'] = data['signup_time'].dt.month
    data['signup_day'] = data['signup_time'].dt.day
    data['signup_hour'] = data['signup_time'].dt.hour
    data['signup_minute'] = data['signup_time'].dt.minute
    data['signup_second'] = data['signup_time'].dt.second


    device_id_map = data['device_id'].value_counts().to_dict()
    data['device_id_count'] = data['device_id'].map(device_id_map)
    source_map = data['source'].value_counts().to_dict()
    data['source_count'] = data['source'].map(source_map)
    browser_map = data['browser'].value_counts().to_dict()
    data['browser_count'] = data['browser'].map(browser_map)
    sex_map = data['sex'].value_counts().to_dict()
    data['sex_count'] = data['sex'].map(sex_map)
    country_map = data['country'].value_counts().to_dict()
    data['country_count'] = data['country'].map(country_map)

    # drop unnecessary columns
    data = data.drop(['signup_time', 'purchase_time', 'device_id', 'user_id', 'source', 'browser', 'sex', 'country', 'purchase_year', 'signup_year', 'ip_address'], axis=1)

    # select the same amount of raw of non fraud as fraud
    fraud_data = data[data['class'] == 1]
    not_fraud_data = data[data['class'] == 0]
    not_fraud_data = not_fraud_data.sample(n=fraud_data.shape[0], random_state=42)
    data_c = pd.concat([fraud_data, not_fraud_data])

    X = data_c.drop('class', axis=1)
    y = data_c['class']
    knn = KNeighborsClassifier(n_neighbors=15)
    sbs = SBS(knn, k_features=1)
    sbs.fit(X, y)
    k_feat = [len(k) for k in sbs.subsets_]
    def plot_function():
        plt.figure(figsize=(15, 10))
        plt.plot(k_feat, sbs.scores_, marker='o', color='grey')
        plt.ylabel('Accuracy')
        plt.xlabel('Number of Features')
        plt.title('Sequential Backward Selection')
        plt.xticks(np.arange(0, 20, 1))
        plt.grid()
    save_plot_as_png(plot_function, 'sequential_backward_selection')

# dendrogram clustering for feature selection
def dendrogram_clustering(data):
    # extract from purchase value year, month, day, hour, minute, second
    data['purchase_year'] = data['purchase_time'].dt.year
    data['purchase_month'] = data['purchase_time'].dt.month
    data['purchase_day'] = data['purchase_time'].dt.day
    data['purchase_hour'] = data['purchase_time'].dt.hour
    data['purchase_minute'] = data['purchase_time'].dt.minute
    data['purchase_second'] = data['purchase_time'].dt.second
    # extract from signup value year, month, day, hour, minute, second
    data['signup_year'] = data['signup_time'].dt.year
    data['signup_month'] = data['signup_time'].dt.month
    data['signup_day'] = data['signup_time'].dt.day
    data['signup_hour'] = data['signup_time'].dt.hour
    data['signup_minute'] = data['signup_time'].dt.minute
    data['signup_second'] = data['signup_time'].dt.second
    device_id_map = data['device_id'].value_counts().to_dict()
    data['device_id_count'] = data['device_id'].map(device_id_map)
    source_map = data['source'].value_counts().to_dict()
    data['source_count'] = data['source'].map(source_map)
    browser_map = data['browser'].value_counts().to_dict()
    data['browser_count'] = data['browser'].map(browser_map)
    sex_map = data['sex'].value_counts().to_dict()
    data['sex_count'] = data['sex'].map(sex_map)
    country_map = data['country'].value_counts().to_dict()
    data['country_count'] = data['country'].map(country_map)
    # drop unnecessary columns
    data = data.drop(['signup_time', 'purchase_time', 'device_id', 'user_id', 'source', 'browser', 'sex', 'country', 'purchase_year', 'signup_year', 'ip_address'], axis=1)
    # balance the data
    fraud_data = data[data['class'] == 1]
    not_fraud_data = data[data['class'] == 0]
    not_fraud_data = not_fraud_data.sample(n=fraud_data.shape[0], random_state=42)
    data_c = pd.concat([fraud_data, not_fraud_data])
    # data that are not fraud
    features = data_c.drop('class', axis=1)
    labels = data_c['class']
    # transpose the data
    features = features.T
    # standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    # calculate the distance matrix
    distance_matrix = pdist(X, metric='euclidean')
    # calculate the linkage matrix
    linkage_matrix = linkage(distance_matrix, method='complete')
    # plot the dendrogram
    def plot_function():
        plt.figure(figsize=(20, 15))
        dendrogram(linkage_matrix, labels=features.index, orientation='left')


        plt.title('Dendrogram of Features')
        plt.ylabel('Features')
        plt.xlabel('Distance')
    save_plot_as_png(plot_function, 'dendrogram_features')

# initialize the python script
if __name__ == '__main__':
    data = load_efraud_dataset('EFraud_Data_Country.csv')
    # data_profiling_report(data)
################# Bi-variate Analysis ####################
#################Numerical-Claas Analysis ################
    # relationship_between_numerical_features_and_label(data)
################# Temporal - Class Analysis ################
    #relationship_between_datetime_features_and_label(data)
    # plot_sign_up_weekly(data)
    # plot_sign_up_daymonth_features(data)
    # plot_sign_up_dateweek_features(data)
    # plot_sign_up_datehours_features(data)
    # plot_purchase_weekly(data)
    # plot_purchase_daymonth_features(data)
    # plot_purchase_dateweek_features(data)
    plot_purchase_datehours_features(data)
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
    #country_browser_relationship(data)
    #signup_purchase_time_relationship(data)
    #purchase_value_purchase_time_relationship(data)
    #signup_device_id_relationship(data)
    #lda_analysis(data)
    #sbs_analysis(data)
    #dendrogram_clustering(data)

