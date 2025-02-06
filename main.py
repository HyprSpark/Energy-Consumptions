# Importing the necessary libraries
import pandas as pd
import sys

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

main(df)