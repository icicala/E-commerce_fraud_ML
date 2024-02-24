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
        # Create a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

        # Box plot
        sns.boxplot(data=data, x='purchase_value', color='grey', ax=ax_box)
        ax_box.set(xlabel='')
        # Histogram with relative frequency
        sns.histplot(data=data, x='purchase_value', bins=30, color='grey', kde=True, ax=ax_hist, stat='density')
        # Draw the mean line
        mean_value = data['purchase_value'].mean()
        ax_hist.axvline(mean_value, color='black', linestyle='dashed', linewidth=2)
        # Access the axes and modify the color of the KDE curve
        ax_hist.lines[0].set_color('red')
        # Add text label for the mean value
        ax_hist.text(mean_value, ax_hist.get_ylim()[1], f'Mean: {mean_value:.2f}', color='black', verticalalignment='top')

        # Customize labels and titles
        plt.suptitle('Distribution of Purchase Values', y=1.00, fontsize=12)
        plt.xlabel('Purchase Values (in USD)')
        plt.ylabel('Relative Frequency')
        # x-axis ticks every 10 units
        plt.xticks(np.arange(0, 161, 10))

        plt.tight_layout()


    save_plot_as_png(plot_function, 'purchase_value')

# Histogram and Box plot for discrete data: age
def histogram_boxplot_discrete_age(data):
    # Plot histogram and box plot for age
    def plot_function():
        # Create a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

        # Box plot
        sns.boxplot(data=data, x='age', color='grey', ax=ax_box)
        ax_box.set(xlabel='')
        # Histogram with relative frequency
        sns.histplot(data=data, x='age', bins=30, color='grey', kde=True, ax=ax_hist, stat='density')
        # Draw the mean line
        mean_value = data['age'].mean()
        ax_hist.axvline(mean_value, color='grey', linestyle='dashed', linewidth=2)
        # Access the axes and modify the color of the KDE curve
        ax_hist.lines[0].set_color('red')
        # Add text label for the mean value
        ax_hist.text(mean_value, ax_hist.get_ylim()[1], f'Mean: {mean_value:.2f}', color='black', verticalalignment='top')

        # Customize labels and titles
        plt.suptitle('Distribution of Age', y=1.00, fontsize=12)
        plt.xlabel('Age (in years)')
        plt.ylabel('Relative Frequency')
        # x-axis ticks every 5 years
        plt.xticks(np.arange(15, 91, 5))
        plt.tight_layout()

    save_plot_as_png(plot_function, 'age')


# Bar chart for class
def bar_chart_class(data):
    def plot_function():
        # Bar plot with relative frequency on the y-axis
        sns.barplot(x=data['class'].value_counts().index, y=data['class'].value_counts(normalize=True), color='grey')
        # Customize labels and titles
        plt.title('Distribution of \"Class\" Variable')
        plt.xlabel('Class')
        # labels for x-axis
        plt.xticks([0, 1], ['Not Fraud', 'Fraud'])
        plt.ylabel('Relative Frequency')

    # Save the plot as PNG using the save_plot_as_png function
    save_plot_as_png(plot_function, 'class')

# Box plot and lineplot for time series with relative frequncy: signup_time per month
def lineplot_boxplot_signup_time(data):
    def plot_function():
        # Create a figure composed of two matplotlib.Axes objects (ax_box and ax_line)
        f, (ax_box, ax_line) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

        # Box plot
        sns.boxplot(data=data, x=data['signup_time'].dt.month, color='grey', ax=ax_box)
        ax_box.set(xlabel='')
        # Line plot with relative frequency
        sns.lineplot(data=data['signup_time'].dt.month.value_counts(normalize=True).sort_index(), marker='o', ax=ax_line, color='grey')
        # Customize labels and titles
        plt.suptitle('Distribution of Sign-Up Time per Month', y=1.00, fontsize=12)
        plt.xlabel('Month')
        plt.xticks(np.arange(1, 9), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'])
        plt.ylabel('Relative Frequency')
        plt.tight_layout()

    save_plot_as_png(plot_function, 'signup_time_month')

# Box plot and lineplot for time series with relative frequncy: purchase_time per month
def lineplot_boxplot_purchase_time(data):
    def plot_function():
        # Create a figure composed of two matplotlib.Axes objects (ax_box and ax_line)
        f, (ax_box, ax_line) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

        # Box plot
        sns.boxplot(data=data, x=data['purchase_time'].dt.month, color='grey', ax=ax_box)
        ax_box.set(xlabel='')
        # Line plot with relative frequency
        sns.lineplot(data=data['purchase_time'].dt.month.value_counts(normalize=True).sort_index(), marker='o', ax=ax_line, color='grey')
        # Customize labels and titles
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


    # Plot bar chart for each categorical column
    for col in categorical_columns:
        def plot_function():
            # Bar plot with relative frequency on the y-axis
            sns.barplot(x=data[col].value_counts().index, y=data[col].value_counts(normalize=True), color='grey')
            # Customize labels and titles
            plt.title(f'Distribution of {col} variable')
            # col name (per category)
            plt.xlabel(col.strip().capitalize())
            plt.ylabel('Relative Frequency')
        save_plot_as_png(plot_function, col)

# Horizontal bar chart for country
def bar_chart_country(data):
    # Calculate total number of transactions
    total_transactions = len(data)

    # Get counts for each country
    country_counts = data['country'].value_counts()

    # Calculate relative frequency for each country
    relative_frequencies = country_counts / total_transactions

    # Sort by relative frequency
    sorted_countries = relative_frequencies.sort_values(ascending=False)

    # Select the top 30 countries
    top_countries = sorted_countries.head(30)

    # Calculate the sum of relative frequencies for other countries
    other_countries_frequency = sorted_countries[30:].sum()

    # Create a DataFrame with top 30 countries and their relative frequencies
    top_countries_df = pd.DataFrame({'Country': top_countries.index, 'Relative Frequency': top_countries.values})

    # Create a DataFrame for the 'Others' category
    others_df = pd.DataFrame({'Country': ['Others'], 'Relative Frequency': [other_countries_frequency]})

    # Concatenate top 30 and 'Others' DataFrames
    combined_dataframe = pd.concat([top_countries_df, others_df])

    def plot_functions():
        # Bar plot with relative frequency on the y-axis
        sns.barplot(x=combined_dataframe['Relative Frequency'], y=combined_dataframe['Country'], color='grey')
        # Customize labels and titles
        plt.title('Distribution of Transactions per Country (Relative Frequency)')
        plt.xlabel('Relative Frequency')
        plt.ylabel('Country')

    save_plot_as_png(plot_functions, 'country')






# Device_ID map to numeric plot Histogram
def histogram_device_id(data):
    # Create a mapping from device_id to unique numeric identifiers
    device_id_mapping = {device_id: idx for idx, device_id in enumerate(data['device_id'].unique())}

    # Map the device_id column to numeric identifiers
    data['device_id_numeric'] = data['device_id'].map(device_id_mapping)

    def plot_function():
        # Histogram with relative frequency on the y-axis
        sns.histplot(data['device_id_numeric'], bins=30, color='black', kde=True, stat='density')

        # Draw the red curve line
        plt.gca().lines[0].set_color('red')
        plt.xlabel('Numeric Device ID')
        plt.ylabel('Relative Frequency')
        plt.title('Distribution of Transactions per Device ID')

    save_plot_as_png(plot_function, 'device_id_numeric')

# Histogram for user_id
def histogram_user_id(data):
    def plot_function():
        # Plot histogram with relative frequency on the y-axis
        sns.histplot(data['user_id'], bins=30, color='black', kde=True, stat='density')
        # Draw the red curve line
        plt.gca().lines[0].set_color('red')
        # Customize labels and titles
        plt.title('Distribution of Transactions per User ID')
        plt.xlabel('User ID')
        plt.ylabel('Relative Frequency')

    save_plot_as_png(plot_function, 'user_id')
# python init
if __name__ == '__main__':
    dataset = load_efraud_dataset('EFraud_Data_Country.csv')
    # Histograms and boxplot for numerical data
    # histogram_boxplot_discrete_purchase_value(dataset)
    # histogram_boxplot_discrete_age(dataset)
    # bar chart for class
    bar_chart_class(dataset)
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

