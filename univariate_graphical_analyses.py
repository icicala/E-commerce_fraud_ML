import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



# load EFraud_Dataset with country
def load_efraud_dataset(filename):
    filename = os.path.join(os.path.dirname(__file__), 'data', filename)
    data = pd.read_csv(filename, parse_dates=['signup_time', 'purchase_time'])
    return data.copy()

def save_plot_as_png(plot_function, plot_name):
    plt.figure(figsize=(15, 12))
    plot_function()
    os.makedirs(os.path.join(os.path.dirname(__file__), 'plot'), exist_ok=True)
    save_file = os.path.join(os.path.dirname(__file__), 'plot', f'{plot_name}_histogram.png')
    plt.savefig(save_file)
    plt.close()
#histogram for discrete data: user_id, purchase_value, age
def histogram_boxplot_discrete(data):
    # Extract numerical columns for histogram
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    # Exclude class and ip_address columns
    numerical_columns = numerical_columns.drop(['class', 'ip_address'])

    # Plot histogram and box plot for each numerical column
    for col in numerical_columns:
        def plot_function():
            # Create a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
            f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

            # Box plot
            sns.boxplot(data=data, x=col, color='grey', ax=ax_box)
            ax_box.set(xlabel='')

            # Histogram
            sns.histplot(data=data, x=col, bins=30, color='black', kde=True, ax=ax_hist)
            # Draw the mean line
            mean_value = data[col].mean()
            ax_hist.axvline(mean_value, color='black', linestyle='dashed', linewidth=2)
            # Access the axes and modify the color of the KDE curve
            ax_hist.lines[0].set_color('red')
            # Add text label for the mean value
            ax_hist.text(mean_value, ax_hist.get_ylim()[1], f'Mean: {mean_value:.2f}', color='black', verticalalignment='top')

            # Customize labels and titles
            plt.suptitle(f'Box plot and Histogram of {col} column', y=1.00, fontsize=12)
            plt.xlabel(col)
            plt.ylabel('Count')

            plt.tight_layout()

        # Save the plot as PNG using the save_plot_as_png function
        save_plot_as_png(plot_function, col)



# Bar chart for class
def bar_chart_class(data):
    def plot_function():
        sns.countplot(data=data, x='class', color='grey')
        # Customize labels and titles
        plt.title('Distribution of Fraudulent and Non-Fraudulent Transactions')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Not Fraud', 'Fraud'])

    # Save the plot as PNG using the save_plot_as_png function
    save_plot_as_png(plot_function, 'class')

# Histogram and Box plot for continuous data: signup_time, purchase_time
def histogram_boxplot_datetime(data):
    # Extract datetime columns
    datetime_columns = ['signup_time', 'purchase_time']

    # Plot histogram and box plot for each datetime column
    for col in datetime_columns:
        def plot_function():
            # Create a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
            f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

            # Box plot
            sns.boxplot(data=data, x=col, color='grey', ax=ax_box)
            ax_box.set(xlabel='')

            # Histogram
            sns.histplot(data=data, x=col, bins=30, color='black', kde=True, ax=ax_hist)
            # Draw the mean line
            mean_value = data[col].mean()
            ax_hist.axvline(mean_value, color='black', linestyle='dashed', linewidth=2)
            # Access the axes and modify the color of the KDE curve
            ax_hist.lines[0].set_color('red')
            # Add text label for the mean value
            ax_hist.text(mean_value, ax_hist.get_ylim()[1], f'Mean: {mean_value:.2f}', color='black', verticalalignment='top')

            # Customize labels and titles
            plt.suptitle(f'Box plot and Histogram of {col} column', y=1.00, fontsize=12)
            plt.xlabel(col)
            plt.ylabel('Count')

            plt.tight_layout()

        # Save the plot as PNG using the save_plot_as_png function
        save_plot_as_png(plot_function, col)

# Bar chart for categorical data: sex, source, browser
def bar_chart_categorical(data):
    # Extract categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    categorical_columns = categorical_columns.drop(['device_id', 'country'])

    # Plot bar chart for each categorical column
    for col in categorical_columns:
        def plot_function():
            sns.countplot(data=data, x=col, color='grey')
            # Customize labels and titles
            plt.title(f'Distribution of {col} column')
            plt.xlabel(col)
            plt.ylabel('Count')

        # Save the plot as PNG using the save_plot_as_png function
        save_plot_as_png(plot_function, col)

# Horizontal bar chart for country
def bar_chart_country(data):
    top_countries = data['country'].value_counts().nlargest(30)
    other_countries = data['country'].value_counts().index[30:]

    # Create a DataFrame with top 30 countries and their counts
    top_countries_df = pd.DataFrame({'Country': top_countries.index, 'Count': top_countries.values})

    # Create a DataFrame for the 'Others' category with the sum of transactions
    others_df = pd.DataFrame({'Country': ['Others'], 'Count': [data[data['country'].isin(other_countries)]['country'].count()]})

    # Concatenate top 30 and 'Others' DataFrames
    combined_dataframe = pd.concat([top_countries_df, others_df])

    def plot_functions():
        sns.barplot(x='Count', y='Country', data=combined_dataframe, color='grey')
        # Customize labels and titles
        plt.title('Top 30 Countries with the Most Transactions')
        plt.xlabel('Count')
        plt.ylabel('Country')
    save_plot_as_png(plot_functions, 'country')

# Device_ID map to numeric plot Histogram
def device_id_histoboxplot(data):
    # Create a mapping from device_id to unique numeric identifiers
    device_id_mapping = {device_id: idx for idx, device_id in enumerate(data['device_id'].unique())}

    # Map the device_id column to numeric identifiers
    data['device_id_numeric'] = data['device_id'].map(device_id_mapping)

    def plot_function():
        # Create a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

        # Box plot
        sns.boxplot(x=data['device_id_numeric'], color='grey', ax=ax_box)
        ax_box.set(xlabel='')

        # Histogram
        sns.histplot(data['device_id_numeric'], bins=30, color='black', kde=True, ax=ax_hist)

        # Draw the red curve line
        ax_hist.lines[0].set_color('red')

        plt.xlabel('Numeric Device ID')
        plt.ylabel('Frequency')
        plt.suptitle('Distribution of Transactions per Device (Numeric IDs)', y=1.0)  # Remove the default super title

        plt.tight_layout()

    save_plot_as_png(plot_function, 'device_id_numeric')
# python init
if __name__ == '__main__':
    dataset = load_efraud_dataset('EFraud_Data_Country.csv')
    # Histograms and boxplot for numerical data
    #histogram_boxplot_discrete(dataset)
    # bar chart for class
    #bar_chart_class(dataset)
    # Histogram and Box plot for datetime data
    #histogram_boxplot_datetime(dataset)
    # Bar chart for categorical data
    #bar_chart_categorical(dataset)
    # Bar chart for country
    #bar_chart_country(dataset)
    # Histogram for device_id
    device_id_histoboxplot(dataset)