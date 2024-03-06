import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np



# load EFraud_Dataset with country
def load_efraud_dataset(filename):
    filename = os.path.join(os.path.dirname(__file__), 'data', filename)
    data = pd.read_csv(filename, parse_dates=['signup_time', 'purchase_time'])
    return data.copy()

def save_plot_as_png(plot_function, plot_name):
    plt.figure(figsize=(15, 7))
    plot_function()
    os.makedirs(os.path.join(os.path.dirname(__file__), 'plots'), exist_ok=True)
    save_file = os.path.join(os.path.dirname(__file__), 'plots', f'{plot_name}_histogram.png')
    plt.savefig(save_file)
    plt.close()
# Histogram and Box plot for discrete data: purchase_value
def histogram_boxplot_discrete_purchase_value(data):

    # Plot histogram and box plot for purchase_value
    def plot_function():
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
        sns.boxplot(data=data, x='purchase_value', color='grey', ax=ax_box)
        ax_box.set(xlabel='')
        sns.histplot(data=data, x='purchase_value', bins=30, color='grey', kde=True, ax=ax_hist, stat='density')
        mean_value = data['purchase_value'].mean()
        ax_hist.axvline(mean_value, color='black', linestyle='dashed', linewidth=2)
        ax_hist.lines[0].set_color('red')
        ax_hist.text(mean_value, ax_hist.get_ylim()[1], f'Mean: {mean_value:.2f}', color='black', verticalalignment='top')
        plt.suptitle('Distribution of Purchase Values', y=1.00, fontsize=12)
        plt.xlabel('Purchase Values (in USD)')
        plt.ylabel('Relative Frequency')
        plt.xticks(np.arange(0, 161, 10))
        plt.tight_layout()


    save_plot_as_png(plot_function, 'purchase_value')

# Histogram and Box plot for discrete data: age
def histogram_boxplot_discrete_age(data):
    def plot_function():
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
        sns.boxplot(data=data, x='age', color='grey', ax=ax_box)
        ax_box.set(xlabel='')
        sns.histplot(data=data, x='age', bins=30, color='grey', kde=True, ax=ax_hist, stat='density')
        mean_value = data['age'].mean()
        ax_hist.axvline(mean_value, color='grey', linestyle='dashed', linewidth=2)
        ax_hist.lines[0].set_color('red')
        ax_hist.text(mean_value, ax_hist.get_ylim()[1], f'Mean: {mean_value:.2f}', color='black', verticalalignment='top')
        plt.suptitle('Distribution of Age', y=1.00, fontsize=12)
        plt.xlabel('Age (in years)')
        plt.ylabel('Relative Frequency')
        plt.xticks(np.arange(15, 91, 5))
        plt.tight_layout()

    save_plot_as_png(plot_function, 'age')


# Bar chart for class
def bar_chart_class(data):
    def plot_function():
        sns.barplot(x=data['class'].value_counts().index, y=data['class'].value_counts(normalize=True), color='grey')
        plt.title('Distribution of \"Class\" Variable')
        plt.xlabel('Class')
        plt.xticks([0, 1], ['Not Fraud', 'Fraud'])
        plt.ylabel('Relative Frequency')
    save_plot_as_png(plot_function, 'class')

# Box plot and lineplot for time series with relative frequncy: signup_time per month
def lineplot_boxplot_signup_time(data):
    def plot_function():
        f, (ax_box, ax_line) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
        sns.boxplot(data=data, x=data['signup_time'].dt.month, color='grey', ax=ax_box)
        ax_box.set(xlabel='')
        sns.lineplot(data=data['signup_time'].dt.month.value_counts(normalize=True).sort_index(), marker='o', ax=ax_line, color='grey')
        plt.suptitle('Distribution of Sign-Up Time per Month', y=1.00, fontsize=12)
        plt.xlabel('Month')
        plt.xticks(np.arange(1, 9), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'])
        plt.ylabel('Relative Frequency')
        plt.tight_layout()

    save_plot_as_png(plot_function, 'signup_time_month')

# Box plot and lineplot for time series with relative frequncy: purchase_time per month
def lineplot_boxplot_purchase_time(data):
    def plot_function():
        f, (ax_box, ax_line) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
        sns.boxplot(data=data, x=data['purchase_time'].dt.month, color='grey', ax=ax_box)
        ax_box.set(xlabel='')
        sns.lineplot(data=data['purchase_time'].dt.month.value_counts(normalize=True).sort_index(), marker='o', ax=ax_line, color='grey')
        plt.suptitle('Distribution of Purchase Time per Month', y=1.00, fontsize=12)
        plt.xlabel('Month')
        plt.xticks(np.arange(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.ylabel('Relative Frequency')
        plt.tight_layout()
    save_plot_as_png(plot_function, 'purchase_time_month')


# Bar chart for categorical data: sex, source, browser
def bar_chart_categorical(data):
    data = data[['source', 'browser', 'sex']].copy()
    categorical_columns = data.columns
    for col in categorical_columns:
        def plot_function():
            sns.barplot(x=data[col].value_counts().index, y=data[col].value_counts(normalize=True), color='grey')
            plt.title(f'Distribution of {col} variable')
            plt.xlabel(col.strip().capitalize())
            plt.ylabel('Relative Frequency')
        save_plot_as_png(plot_function, col)

# Horizontal bar chart for country
def bar_chart_country(data):

    total_transactions = len(data)
    country_counts = data['country'].value_counts()
    relative_frequencies = country_counts / total_transactions
    sorted_countries = relative_frequencies.sort_values(ascending=False)
    top_countries = sorted_countries.head(30)
    other_countries_frequency = sorted_countries[30:].sum()
    top_countries_df = pd.DataFrame({'Country': top_countries.index, 'Relative Frequency': top_countries.values})
    others_df = pd.DataFrame({'Country': ['Others'], 'Relative Frequency': [other_countries_frequency]})
    combined_dataframe = pd.concat([top_countries_df, others_df])

    def plot_functions():
        sns.barplot(x=combined_dataframe['Relative Frequency'], y=combined_dataframe['Country'], color='grey')
        plt.title('Distribution of Transactions per Country (Relative Frequency)')
        plt.xlabel('Relative Frequency')
        plt.ylabel('Country')
    save_plot_as_png(plot_functions, 'country')

# Device_ID map to numeric plot Histogram
def histogram_device_id(data):
    device_id_mapping = {device_id: idx for idx, device_id in enumerate(data['device_id'].unique())}
    data['device_id_numeric'] = data['device_id'].map(device_id_mapping)

    def plot_function():
        sns.histplot(data['device_id_numeric'], bins=30, color='black', kde=True, stat='density')
        plt.gca().lines[0].set_color('red')
        plt.xlabel('Numeric Device ID')
        plt.ylabel('Relative Frequency')
        plt.title('Distribution of Transactions per Device ID')

    save_plot_as_png(plot_function, 'device_id_numeric')

# Histogram for user_id
def histogram_user_id(data):
    def plot_function():
        sns.histplot(data['user_id'], bins=30, color='black', kde=True, stat='density')
        plt.gca().lines[0].set_color('red')
        plt.title('Distribution of Transactions per User ID')
        plt.xlabel('User ID')
        plt.ylabel('Relative Frequency')
    save_plot_as_png(plot_function, 'user_id')
if __name__ == '__main__':
    dataset = load_efraud_dataset('EFraud_Data_Country.csv')
    # Histograms and boxplot for numerical data
    histogram_boxplot_discrete_purchase_value(dataset)
    # histogram_boxplot_discrete_age(dataset)
    # bar chart for class
    # bar_chart_class(dataset)
    # Histogram and Box plot for datetime data
    #lineplot_boxplot_signup_time(dataset)
    # lineplot_boxplot_purchase_time(dataset)
    # Bar chart for categorical data
    # bar_chart_categorical(dataset)
    # Bar chart for country
    #bar_chart_country(dataset)
    # histogram for user_id
    #histogram_user_id(dataset)
    # Device_ID map to numeric plot Histogram
    #histogram_device_id(dataset)

