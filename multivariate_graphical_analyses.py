from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import os


def load_efraud_dataset(filename):
    filename = os.path.join(os.path.dirname(__file__), 'data', filename)
    data = pd.read_csv(filename, parse_dates=['signup_time', 'purchase_time'])
    return data.copy()
# correlation plot- heatmap





#initialize the python script
if __name__ == '__main__':
    data = load_efraud_dataset('EFraud_Data_Country.csv')