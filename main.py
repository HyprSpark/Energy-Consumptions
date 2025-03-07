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


### PREDICTION MODELS ###


# Making predictions using the Naive algorithm, most basic method
def naive_model(train, test):
    # Use the last observed value for all future predictions
    last_value = train['Total_feeder'].iloc[-1]
    
    predictions = np.full(len(test), last_value) # Filling the predictions with the last value

    # Calculate Mean Squared Error
    mse = mean_squared_error(test['Total_feeder'], predictions)
    print(f"Naive Model MSE: {mse}")
    
    return predictions, mse

# Making predictions using the Seasonal Naive algorithm to try and get a better prediction with trends
def seasonal_naive_model(train, test, period=24):
    test = test.copy()

    # Shift data by 'period' steps
    test['Predictions'] = train['Total_feeder'].shift(period).iloc[-len(test):].values

    # Fixing the FutureWarning by assigning properly
    test.loc[:, 'Predictions'] = test['Predictions'].fillna(train['Total_feeder'].mean())

    # Calculate Mean Squared Error
    mse = mean_squared_error(test['Total_feeder'], test['Predictions'])
    print(f"Seasonal Naive Model MSE: {mse}")

    return test['Predictions'].values, mse

# Making predictions using the Linear Regression algorithm 
def linear_regression(train, test):
    # Ensure we are not modifying the original datasets
    train = train.copy()
    test = test.copy()
    
    # Check if 'Time' is in datetime format
    if not np.issubdtype(train['Time'].dtype, np.datetime64): # Checking if the data type is datetime
        train['Time'] = pd.to_datetime(train['Time']) # Converting the data type to datetime
        test['Time'] = pd.to_datetime(test['Time'])

    # Extracting additional time-based features
    train['DayOfWeek'] = train['Time'].dt.dayofweek
    train['Hour'] = train['Time'].dt.hour
    train['Day'] = train['Time'].dt.day

    # Extracting additional time-based features
    test['DayOfWeek'] = test['Time'].dt.dayofweek
    test['Hour'] = test['Time'].dt.hour
    test['Day'] = test['Time'].dt.day

    # Prepare features and target
    X_train = train[['DayOfWeek', 'Hour', 'Day']] # Features
    y_train = train['Total_feeder'] # Target

    X_test = test[['DayOfWeek', 'Hour', 'Day']] # Features
    y_test = test['Total_feeder'] # Target

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
    
    return predictions, mse

# Making predictions using the ARIMA model
def ARIMA_test(test, train):

  # Set the index to the 'Time' column
  train = train.set_index('Time')
  test = test.set_index('Time')

  # Resample the data to hourly
  train = train.asfreq('h')
  test = test.asfreq('h')

  # Create the ARIMA model
  model_train = ARIMA(train['Total_feeder'], order=(12, 1, 1))
  model_train_fit = model_train.fit()

  # Forecast the data
  test_forecast = model_train_fit.forecast(steps=30)

  # Create a series for the forecast
  test_forecast_series = pd.Series(test_forecast)

  # Calculate the Mean Squared Error
  ARMIAMSE = mean_squared_error(test['Total_feeder'][-30:], test_forecast_series)
  print(f'ARIMA Model MSE: {ARMIAMSE}')

  return ARMIAMSE, test_forecast_series

### DATA ANALYSIS ###

# Data analysis function to compare the models using the RMSE
def dataAnalysis(predictionsLinear, predictionsNaive, predictionsSeasonalNaive, linearMSE, naiveMSE, seasonalMSE, ARIMAMSE):
  
  # Calculate the mean consumption of the dataset
  mean_consumption = df['Total_feeder'].mean()
  print(mean_consumption)

  linear_rmse = np.sqrt(linearMSE) # Root Mean Squared Error for Linear Regression
  naive_rmse = np.sqrt(naiveMSE) # Root Mean Squared Error for Naive Model
  seasonal_rmse = np.sqrt(seasonalMSE) # Root Mean Squared Error for Seasonal Naive Model
  ARIMA_rmse = np.sqrt(ARIMAMSE) # Root Mean Squared Error for ARIMA Model
  
  # Print the RMSE for each model
  print('\n\nRoot Mean Squared Error (RMSE) for each model:') 
  print(f"Linear Regression RMSE: {linear_rmse}")
  print(f"Naive Model RMSE: {naive_rmse}")
  print(f"Seasonal Naive Model RMSE: {seasonal_rmse}")
  print('\n\n')

  # Calculate the RMSE percentage for each model
  rmse_linear_pct = round((linear_rmse / mean_consumption) * 100,2)
  rmse_naive_pct = round((naive_rmse / mean_consumption) * 100,2)
  rmse_seasonal_naive_pct = round((seasonal_rmse / mean_consumption) * 100, 2)
  rmse_ARIMA = round((ARIMA_rmse / mean_consumption) * 100,2)

  # Print the RMSE error percentage for each model
  print(f'Error from main;\nLinear error (%): {rmse_linear_pct}% \nNaive error (%) {rmse_naive_pct}% \nSeasonal naive error (%): {rmse_seasonal_naive_pct}%\nRMSE error for ARIMA model: {rmse_ARIMA}%')

## PLOTTING FUNCTIONS ##

def plot_naive_model(train, test, predictionsNaive):

  plt.figure(figsize=(10,6))
  #plt.plot(train['Total_feeder'], label='Training Data')
  plt.plot(test['Total_feeder'], label='Testing Data', color='orange')
  plt.plot(test.index, predictionsNaive, label='Naive Forecast', color='red')

  plt.title('Naive Model Forecast')
  plt.xlabel('Date')
  plt.ylabel('Total Feeder')
  plt.legend()
  plt.show()

def plot_seasonal_naive_model(train, test, predictionsSeasonalNaive):
   
  plt.figure(figsize=(10,6))
  #plt.plot(train['Total_feeder'], label = 'Training Data')
  plt.plot(test['Total_feeder'], label = 'Testing Data', color = 'orange')
  plt.plot(test.index, predictionsSeasonalNaive, label = 'Seasonal Naive Forecast', color = 'red')

  plt.title('Seasonal Naive Model Forecast')
  plt.xlabel('Date')
  plt.ylabel('Total Feeder')
  plt.legend()
  plt.show()

def plot_linear_regression_model(train, test, predictionsLinear):
   
  plt.figure(figsize=(10,6))
  #plt.plot(train['Total_feeder'], label='Training Data')
  plt.plot(test['Total_feeder'], label='Testing Data', color='orange')
  plt.plot(test.index, predictionsLinear, label='Linear Regression Forecast', color='red')

  plt.title('Linear Regression Model Forecast')
  plt.xlabel('Date')
  plt.ylabel('Total Feeder')
  plt.legend()
  plt.show()

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


### MAIN FUNCTION ###

def main(df):
  # Converting the string into a datetime format
  df = time_period(df)

  full_weeks = week_grouping(df)
  week_list = list(full_weeks.keys())

  train, test = train_test_split(week_list, full_weeks)


  predictionsLinear, linearMSE = linear_regression(train, test) # Making predictions using the Linear Regression algorithm

  predictionsNaive, naiveMSE = naive_model(train, test) #   Making predictions using the Naive algorithm, most basic method

  predictionsSeasonalNaive, seasonalMSE = seasonal_naive_model(train, test) # Making predictions using the Seasonal Naive algorithm to try and get a better prediction with trends

  ARIMAMSE, test_forecast_series = ARIMA_test(test, train) # ARIMA test

  #dataAnalysis(predictionsLinear, predictionsNaive, predictionsSeasonalNaive, linearMSE, naiveMSE, seasonalMSE, ARIMAMSE) # Data analysis function to compare the models using the RMSE

  # Print the predictions for each model
  print(predictionsLinear, '\n\n')
  print(predictionsNaive, '\n\n')
  print(predictionsSeasonalNaive, '\n\n')

  # Plot the Naive Model
  plot_naive_model(train, test, predictionsNaive)

  # Plot the Seasonal Naive Model
  plot_seasonal_naive_model(train, test, predictionsSeasonalNaive)

  # Plot the Linear Regression Model
  plot_linear_regression_model(train, test, predictionsLinear)

  # Plot the ARIMA Model
  plot_ARIMA_model(train, test, test_forecast_series)

  


 
  
main(df) # Calling the main function

  
