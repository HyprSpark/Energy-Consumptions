# Importing the necessary libraries
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
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


def custom_train_test_split(filtered_days):
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
  
def scatter_plot(train):
  sns.scatterplot(x=train['Time'], y=train['Total_feeder'])
  plt.show()

def naive_bayes(train, test):
  model = GaussianNB()

  # Training the model
# Convert datetime to numerical values
  model.fit(train['Time'].to_frame(), train['Total_feeder'].to_frame())






def main(df):
  # Converting the string into a datetime format
  df = time_period(df)

  # Grouping the data by days
  grouped_df = day_grouping(df)

  # Filtering the data for the desired days that include any kind of power usage
  filtered_days = filtered_empty_days(grouped_df)

  

  train, test = custom_train_test_split(filtered_days)

  print(filtered_days.sample(25))
  predictions = naive_bayes(train, test)
  print(predictions)

  naive_bayes(train, test)


  #plot_creation(train, test)

main(df)
