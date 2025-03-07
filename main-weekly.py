# Importing the necessary libraries
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA



# Attempting to read the data file
try:
  df = pd.read_csv('DataSheets/Energy Advice and Cunsumption Prediction Dataset.csv', names=['Time', 'Total_feeder'], skiprows=1) # Giving the columns names and skipping the first row

except FileNotFoundError: # Checking if the file is not found
  # If the file is not found, print an error message and exit the program
  print('File name incorrect or not found')
  sys.exit() # Exits the program,.-l


### DATA PREPROCESSING ###


# Converting the string into a datetime format
def time_period(df):
  df['Time'] = pd.to_datetime(df['Time']) # saves the time column as a datetime object
  
  return df

# Grouping the data by Weeks
def week_grouping(df):
  grouped_df = df.groupby([pd.Grouper(key='Time', freq='W-MON')]) # Grouping the data by weeks

  return grouped_df

# Filtering the data for the desired weeks that include any kind of power usage
def week_grouping(df):
    grouped = df.groupby(pd.Grouper(key='Time', freq='W-MON'))  # Group by weeks (starting Monday)
    
    filtered_groups = {
        name: group for name, group in grouped
        if group['Time'].dt.date.nunique() == 7 and not group['Total_feeder'].isna().any()
    }
    
    return filtered_groups

def train_test_split(week_list, full_weeks):
    train_weeks = week_list[:6] # Get the first 6 weeks (for training)
    test_weeks = week_list[6:7]  # Get the remaining weeks (for testing)

    # Extract the data for these weeks from the full_weeks dictionary
    train_data = pd.concat([full_weeks[week] for week in train_weeks], axis=0)
    test_data = pd.concat([full_weeks[week] for week in test_weeks], axis=0)

    print('Train data:', train_data)
    print('Test data:', test_data)

    return train_data, test_data
    
### MAIN FUNCTION ###

def main(df):
  # Converting the string into a datetime format
    df = time_period(df)



  # Grouping the data by days
    full_weeks = week_grouping(df)
    print('Total number of weeks:', len(full_weeks))
    #print('Full weeks:', full_weeks)
    week_list = list(full_weeks.keys())

    #print('Week list:', week_list)

    train, test = train_test_split(week_list, full_weeks)



  # Filtering the data for the desired days that include any kind of power usage

    
    

  


 
  
main(df) # Calling the main function

  
