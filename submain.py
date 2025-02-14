# Importing the necessary libraries
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Attempting to read the data file
try:
  
  df = pd.read_csv('DataSheets/Energy Advice and Cunsumption Prediction Dataset.csv', names=['Time', 'Total_feeder'], skiprows=1)

except FileNotFoundError: # Checking if the file is not found
  
  # If the file is not found, print an error message and exit the program
  print('File name incorrect or not found')
  sys.exit()

# Converting the string into a datetime format
def time_period(df):
  df['Time'] = pd.to_datetime(df['Time'])
  
  return df



# Grouping the data by days
def day_grouping(df):
  grouped_df = df.groupby(df['Time'].dt.date) # Grouping the data by days

  return grouped_df

# Filtering the data for the desired days that include any kind of power usage
def filtered_empty_days(grouped_df):
  filtered_days = grouped_df.filter(lambda x: x['Total_feeder'].notna().any()) 
  filtered_days['Total_feeder'] = filtered_days['Total_feeder'].fillna(0) # Filling the NaN values with 0

  return filtered_days


# Splitting the data into training and testing sets based on a custom date
def custom_train_test_split(filtered_days):
  train = filtered_days[filtered_days['Time'] <= pd.to_datetime('2015-11-05', format='%Y-%m-%d')] # Training set
  test = filtered_days[filtered_days['Time'] > pd.to_datetime('2015-11-05', format='%Y-%m-%d')] # Testing set

  return train, test
  
def plot_creation(train, test):
  sns.set_theme()
  plt.figure(figsize=(16,8))
  plt.ylabel('Energy Consumption')
  plt.xlabel('Time')
    
  plt.plot(train['Time'], train['Total_feeder'], label='Train Data', color='black')
  plt.plot(test['Time'], test['Total_feeder'], label='Test Data', color='red')
  plt.legend()
             
  plt.title('Energy Consumption over Time')
  plt.xtricks(45)
  plt.savefig('/workspaces/Energy-Consumptions/assets/energy_consumption_plot_2.png')
  
def scatter_plot(train):
  plt.figure(figsize=(15,8))
  sns.scatterplot(x=train['Time'], y=train['Total_feeder'])
  plt.xtricks(45)
  plt.savefig('/workspaces/Energy-Consumptions/assets/scatter_plot.png')

def naive_bayes(train, test):
  model = statsforcast()

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
  
  plot_creation(train, test)  
  scatter_plot(train) 

main(df)
