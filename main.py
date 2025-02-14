# Importing the necessary libraries
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Attempting to read the data file
try:
  df = pd.read_csv('DataSheets/Energy Advice and Cunsumption Prediction Dataset.csv', names=['Time', 'Total_feeder'], skiprows=1) # Giving the columns names and skipping the first row

except FileNotFoundError: # Checking if the file is not found
  # If the file is not found, print an error message and exit the program
  print('File name incorrect or not found')
  sys.exit() # Exits the program


# Converting the string into a datetime format
def time_period(df):
  df['Time'] = pd.to_datetime(df['Time']) # saves the time column as a datetime object
  
  return df


# Grouping the data by days
def day_grouping(df):
  grouped_df = df.groupby(df['Time'].dt.date) # Grouping the data by days

  return grouped_df


# Filtering the data for the desired days that include any kind of power usage
def filtered_empty_days(grouped_df):
  filtered_days = grouped_df.filter(lambda x: x['Total_feeder'].notna().any()) #filters using notna() function with any() function
  filtered_days['Total_feeder'] = filtered_days['Total_feeder'].fillna(0) # Filling the NaN values with 0

  return filtered_days


# Splitting the data into training and testing sets based on a custom date
def custom_train_test_split(filtered_days):
  train = filtered_days[filtered_days['Time'] <= pd.to_datetime('2015-11-05', format='%Y-%m-%d')] # Training set
  test = filtered_days[filtered_days['Time'] > pd.to_datetime('2015-11-05', format='%Y-%m-%d')] # Testing set

  return train, test


# Making predictions using the Naive Bayes algorithm
#def naive_bayes(train, test):
  #from sklearn.naive_bayes import GaussianNB


  # Convert datetime to numerical values
  #model.fit(train['Time'].to_frame(), train['Total_feeder'].to_frame()) # Fitting the model


def naive_model(train, test):
    # Use the last observed value for all future predictions
    last_value = train['Total_feeder'].iloc[-1]
    
    predictions = np.full(len(test), last_value)

    # Calculate Mean Squared Error
    mse = mean_squared_error(test['Total_feeder'], predictions)
    print(f"Naive Model MSE: {mse}")
    
    return predictions

def seasonal_naive_model(train, test, period=24):
    """
    Uses the value from 'period' hours ago as prediction.
    If period is too large, try reducing it (e.g., 12 hours).
    """
    test = test.copy()

    # Shift data by 'period' steps
    test['Predictions'] = train['Total_feeder'].shift(period).iloc[-len(test):].values

    # Fixing the FutureWarning by assigning properly
    test.loc[:, 'Predictions'] = test['Predictions'].fillna(train['Total_feeder'].mean())

    # Calculate Mean Squared Error
    mse = mean_squared_error(test['Total_feeder'], test['Predictions'])
    print(f"Fixed Seasonal Naive Model MSE: {mse}")

    return test['Predictions'].values




def linear_regression(train, test):
    # Ensure we are not modifying the original datasets
    train = train.copy()
    test = test.copy()
    
    # Check if 'Time' is in datetime format
    if not np.issubdtype(train['Time'].dtype, np.datetime64):
        train['Time'] = pd.to_datetime(train['Time'])
        test['Time'] = pd.to_datetime(test['Time'])

    # Extracting additional time-based features
    train['DayOfWeek'] = train['Time'].dt.dayofweek
    train['Hour'] = train['Time'].dt.hour
    train['Month'] = train['Time'].dt.month
    train['Day'] = train['Time'].dt.day

    test['DayOfWeek'] = test['Time'].dt.dayofweek
    test['Hour'] = test['Time'].dt.hour
    test['Month'] = test['Time'].dt.month
    test['Day'] = test['Time'].dt.day

    # Prepare features and target
    X_train = train[['DayOfWeek', 'Hour', 'Month', 'Day']]
    y_train = train['Total_feeder']

    X_test = test[['DayOfWeek', 'Hour', 'Month', 'Day']]
    y_test = test['Total_feeder']

    # Drop NaN values if any
    X_train, y_train = X_train.dropna(), y_train.dropna()
    X_test, y_test = X_test.dropna(), y_test.dropna()

    # Initialize and train Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate model performance (using Mean Squared Error)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    
    return predictions



def main(df):
  # Converting the string into a datetime format
  df = time_period(df)

  # Grouping the data by days
  grouped_df = day_grouping(df)

  # Filtering the data for the desired days that include any kind of power usage
  filtered_days = filtered_empty_days(grouped_df)

  # Splitting the data into training and testing sets based on a custom date
  train, test = custom_train_test_split(filtered_days)


  predictionsLinear = linear_regression(train, test)

  predictionsNaive = naive_model(train, test)

  predictionsSeasonalNaive = seasonal_naive_model(train, test)

  print(predictionsLinear, '\n\n')
  print(predictionsNaive, '\n\n')
  print(predictionsSeasonalNaive, '\n\n')




main(df)
