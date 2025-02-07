# Importing the necessary libraries
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
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
  
  
def main(df):
  # Converting the string into a datetime format
  df = time_period(df)

  # Grouping the data by days
  grouped_df = day_grouping(df)

  # Filtering the data for the desired days that include any kind of power usage
  filtered_days = filtered_empty_days(grouped_df)

  print(filtered_days.sample(25))

  sns.set_theme()
  plt.ylabel('Energy Consumption')
  plt.xlabel('Time')
  
  filtered_days.set_index('Time', inplace=True)
  plt.plot(filtered_days.index, filtered_days['Total_feeder'])
  plt.savefig()

main(df)
my_days= list(range())

fig,ax = plt.subplots()
ax.plot(df['date'], df['date'])
ax.set_xlable('Date')
ax.set_ylable('Energy Consumption')
ax.axvspan(**,**,colour = '#808080',alpha=0.2)

plt.xtricks(np.arrange(**,**,**),my_days)
fig.autofnt_xdate()
fig.tight_layout()

plt.savefig('



def train_test_split(filtered_days):
  # Splitting the data into training and testing sets
  train = filtered_days[:int(0.8*(len(filtered_days)))]
  test = filtered_days[int(0.8*(len(filtered_days))):]

  return train, test
  



