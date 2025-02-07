# Importing the necessary libraries
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
import numpy as np


# Attempting to read the data file
try:
  
  df = pd.read_csv('DataSheets/Energy Advice and Cunsumption Prediction Dataset.csv', names=['Time', 'Total_feeder'], skiprows=1)
except FileNotFoundError:
  
  # If the file is not found, print an error message and exit the program
  print('File name incorrect or not found')
  sys.exit()

# Converting the string into a datetime format
def time_period(df):
  df['Time'] = pd.to_datetime(df['Time'])
  return df


# Grouping the data by days
def day_grouping(df):
  # Splitting the data into days
  grouped_df = df.groupby(df['Time'].dt.date)

  return grouped_df


def filtered_empty_days(grouped_df):
  # Filtering the data for the desired days
  filtered_days = grouped_df.filter(lambda x: x['Total_feeder'].notna().any())
  filtered_days['Total_feeder'] = filtered_days['Total_feeder'].fillna(0)
  return filtered_days

def test(df, filtered_days):
  print("Total empty cells:", df.isnull().sum().sum(), "\n\n")
  print(df.shape)

  
  print('Total empty cells:', filtered_days.isnull().sum().sum(), "\n\n")
  print(filtered_days.shape)

def train_test_split(filtered_days):
  # Splitting the data into training and testing sets
  train = filtered_days[filtered_days['Time'] <= pd.to_datetime('2008-09-23', format='%Y-%m-%d')]
  test = filtered_days[filtered_days['Time'] > pd.to_datetime('2008-09-23', format='%Y-%m-%d')]

  return train, test
  
def plot_creation(train, test):
  sns.set_theme()
  plt.ylabel('Energy Consumption')
  plt.xlabel('Time')
    
  train.set_index('Time', inplace=True)
  test.set_index('Time', inplace=True)
    
  plt.plot(train.index, train['Total_feeder'], label='Train', color='black')
  plt.plot(test.index, test['Total_feeder'], label='Test', color='red')
  plt.legend()
  plt.savefig('/workspaces/Energy-Consumptions/assets/energy_consumption_plot.png')
  
def naive_bayes(train):
    model = GaussianNB()

    # Feature Engineering
    train['Hour'] = train['Time'].dt.hour
    train['DayOfWeek'] = train['Time'].dt.dayofweek
    train['Prev_Consumption'] = train['Total_feeder'].shift(1)

    # Handle missing values by filling them with the previous value
    train.bfill(inplace=True)

    # Features and target
    train_x = train[['Hour', 'DayOfWeek', 'Prev_Consumption']].values
    train_y = train['Total_feeder'].values

    # Check if training data is empty
    if len(train_x) == 0 or len(train_y) == 0:
        print("Error: No valid training data available.")
        sys.exit()

    # Train the model
    model.fit(train_x, train_y)
    return model





def main(df):
  # Converting the string into a datetime format
  df = time_period(df)

  # Grouping the data by days
  grouped_df = day_grouping(df)

  # Filtering the data for the desired days that include any kind of power usage
  filtered_days = filtered_empty_days(grouped_df)

  print(filtered_days.sample(25))


  train, test = train_test_split(filtered_days)

  naive_bayes(train)

  #plot_creation(train, test)

main(df)

