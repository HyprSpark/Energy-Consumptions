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
import warnings
import itertools
warnings.filterwarnings("ignore")




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

  train = filtered_days[(filtered_days['Time'] >= pd.to_datetime('2013-04-24', format='%Y-%m-%d')) & (filtered_days['Time'] <= pd.to_datetime('2013-05-20', format='%Y-%m-%d'))] # Training set
  test = filtered_days[(filtered_days['Time'] > pd.to_datetime('2013-04-30', format='%Y-%m-%d')) & (filtered_days['Time'] <= pd.to_datetime('2013-05-21', format='%Y-%m-%d'))] # Testing set

  return train, test


### PREDICTION MODELS ###



# Making predictions using the ARIMA model
def ARIMA_test(test, train):

  # Set the index to the 'Time' column
  train = train.set_index('Time')
  test = test.set_index('Time')

  # Resample the data to hourly
  train = train.asfreq('h')
  test = test.asfreq('h')

  # Create the ARIMA model
  model_train = ARIMA(train['Total_feeder'], order=(2,1,0))
  model_train_fit = model_train.fit()

  # Forecast the data
  test_forecast = model_train_fit.forecast(steps=30)

  # Create a series for the forecast
  test_forecast_series = pd.Series(test_forecast, index=test.index)

  # Calculate the Mean Squared Error
  ARMIAMSE = mean_squared_error(test['Total_feeder'], test_forecast_series)
  print(f'ARIMA Model MSE: {ARMIAMSE}')

  return ARMIAMSE, test_forecast_series

### DATA ANALYSIS ###

# Data analysis function to compare the models using the RMSE

def plot_ARIMA_model(train, test, test_forecast_series):

  # Set the index to the 'Time' column
  train = train.set_index('Time')
  test = test.set_index('Time')

  # Resample the data to hourly
  train = train.asfreq('h')
  test = test.asfreq('h')
    
  # Plot the ARIMA forecast
  plt.figure(figsize=(10,6))
  plt.plot(train['Total_feeder'], label='Training Data')
  plt.plot(test['Total_feeder'], label='Testing Data', color='orange')
  plt.plot(test_forecast_series, label='ARIMA Forecast', color='red')  

  plt.title('ARIMA Model Forecast')
  plt.xlabel('Date')
  plt.ylabel('Total Feeder')
  plt.legend()
  plt.show()

'''
def evaluate_arima_model(x, arima_order):
  train = x[(x['Time'] >= pd.to_datetime('2013-05-13', format='%Y-%m-%d')) & (x['Time'] <= pd.to_datetime('2013-05-20', format='%Y-%m-%d'))] # Training set
  test = x[(x['Time'] > pd.to_datetime('2013-05-20', format='%Y-%m-%d')) & (x['Time'] <= pd.to_datetime('2013-05-21', format='%Y-%m-%d'))] # Testing set

  history = [x for x in train]

  predictions = list()


  for t in range(len(test)):
    model = ARIMA(history, order=arima_order)
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(test[t])

  rmse = np.sqrt(mean_squared_error(test, predictions))
  return rmse


def evaluate_models(dataset, p_values, d_values, q_values):
  best_score, best_cfg = float("inf"), None
  for p in p_values:
    for d in d_values:
      for q in q_values:
        order = (p,d,q)
        try:
          rmse = evaluate_arima_model(dataset, order)
          if rmse < best_score:
            best_score, best_cfg = rmse, order
          print(f'ARIMA {order} MSE= {rmse} ')
        except:
          continue
  print(f'Best ARIMA {best_cfg} MSE= {best_score}')
'''

def evaluate_arima_model(train, test, order):
    try:
        train = train.set_index('Time')
        test = test.set_index('Time')

        train = train.asfreq('h')
        test = test.asfreq('h')

        # Fit ARIMA model
        model = ARIMA(train['Total_feeder'], order=order)
        model_fit = model.fit()

        # Forecasting test set
        forecast = model_fit.forecast(steps=len(test))

        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(test['Total_feeder'], forecast))

        return rmse, forecast
    except Exception as e:
        print(f"Error with ARIMA{order}: {e}")
        return float("inf"), None


def evaluate_models(train, test, p_values, d_values, q_values):
    best_score, best_order, best_forecast = float("inf"), None, None
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    rmse, forecast = evaluate_arima_model(train, test, order)
                    if rmse < best_score:
                        best_score, best_order, best_forecast = rmse, order, forecast
                    print(f"ARIMA{order} RMSE={rmse:.3f}")
                except Exception as e:
                    print(f"Skipping ARIMA{order} due to error: {e}")
                    continue

    print(f"Best ARIMA{best_order} RMSE={best_score:.3f}")
    return best_order, best_score, best_forecast







### MAIN FUNCTION ###

def main(df):
  # Converting the string into a datetime format
  df = time_period(df)

  # Grouping the data by days
  grouped_df = day_grouping(df)

  # Filtering the data for the desired days that include any kind of power usage
  filtered_days = filtered_empty_days(grouped_df)

  # Splitting the data into training and testing sets based on a custom date
  train, test = custom_train_test_split(filtered_days)

  ARIMAMSE, test_forecast_series = ARIMA_test(test, train) # ARIMA test

  # Plot the ARIMA Model
  p_values = range(0, 14, 2)
  d_values = range(0, 4)
  q_values = range(0, 4)
    
  best_order, best_rmse, best_forecast = evaluate_models(train, test, p_values, d_values, q_values)

    # Convert forecast to pandas Series
  best_forecast_series = pd.Series(best_forecast, index=test.set_index('Time').index)

    # Plot Best ARIMA Model
  plot_ARIMA_model(train, test, best_forecast_series)



 
  
main(df) # Calling the main function

