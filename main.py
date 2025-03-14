# Description: This program reads a dataset and predicts the energy consumption using different models such as Naive, Seasonal Naive, Linear Regression and ARIMA.
## IMPORTING LIBRARIES ##


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
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Ignoring the warnings set when testing the ARIMA model
warnings.filterwarnings("ignore")


## DATA READING ##


# Attempting to read the data file
try:
  df = pd.read_csv('DataSheets/Energy Advice and Cunsumption Prediction Dataset.csv', names=['Time', 'Total_feeder'], skiprows=1) # Giving the columns names and skipping the first row

except FileNotFoundError: # Checking if the file is not found
  # If the file is not found, print an error message and exit the program
  print('File name incorrect or not found')
  sys.exit() # Exits the program,.-l


## DATA PREPROCESSING ##


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

# Splitting the data into training and testing data
def train_test_split(week_list, full_weeks, i):
    

    week_name = list(full_weeks.keys())[i]  #  Selecting 'i' week
    week_df = full_weeks[week_name]  # Get the corresponding DataFrame

    week_df['Time'] = pd.to_datetime(week_df['Time']) # Converting the 'Time' column to datetime

    days = list(week_df.groupby(week_df['Time'].dt.normalize()))  # Group by date
    train_data = [group for _, group in days[:6]]  # First 6 days for training
    test_data = [group for _, group in days[6:7]]  # Last day for testing

    train_data, test_data = pd.concat(train_data), pd.concat(test_data) # Concatenating the data 

    return  train_data, test_data


## PREDICTION MODELS ##


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
    test = test.copy() # Creating a copy of the test data

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

# Evaluate an ARIMA model for a given order (p,d,q)
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
    
# Evaluate an ARIMA model for a given order (p,d,q)
def evaluate_models(train, test):
    
    # Define the p, d, q values to test
    p_values = range(8, 12, 2)
    d_values = range(0, 4)
    q_values = range(0, 4)
    
    best_score, best_order, best_forecast = float("inf"), None, None # Initialize variables to store the best model
    total_tests = len(p_values) * len(d_values) * len(q_values)  # Total number of ARIMA models being tested

    i = 1  # Initialize counter for progress tracking

    # Loop through all possible combinations of p, d, q values
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)

                # Evaluate the ARIMA model
                try:
                    rmse, forecast = evaluate_arima_model(train, test, order)

                    # Update the best model if the current model is better
                    if rmse < best_score:
                        best_score, best_order, best_forecast = rmse, order, forecast

                # Skip the current model if an error occurs
                except Exception as e:
                    continue
                
                # Print progress
                print(f"ORDER TESTED {i}/{total_tests}", end='\r')
                i += 1  # Increment the counter

    # Print the best model
    print(f"Best ARIMA{best_order} RMSE={best_score:.3f}")

    return best_order, best_score, best_forecast

# Making predictions using the ARIMA model
def ARIMA_test(test, train, best_order):

  # Set the index to the 'Time' column
  train = train.set_index('Time')
  test = test.set_index('Time')

  # Resample the data to hourly
  train = train.asfreq('h')
  test = test.asfreq('h')

  # Create the ARIMA model
  model_train = ARIMA(train['Total_feeder'], order=(best_order))
  model_train_fit = model_train.fit()

  test_forecast = model_train_fit.forecast(steps=len(test))  # Forecast the data

  test_forecast_series = pd.Series(test_forecast) # Create a series for the forecast

  # Calculate the Mean Squared Error
  ARMIAMSE = mean_squared_error(test['Total_feeder'], test_forecast_series)
  print(f'ARIMA Model MSE: {ARMIAMSE}')

  return ARMIAMSE, test_forecast_series

## DATA ANALYSIS ##

# Data analysis function to compare the models using the RMSE
def dataAnalysis(predictionsLinear, predictionsNaive, predictionsSeasonalNaive, linearMSE, naiveMSE, seasonalMSE, ARIMAMSE):
  
  # Calculate the mean consumption of the dataset
  mean_consumption = df['Total_feeder'].mean()

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


# Plotting the naive model
def plot_naive_model(test, predictionsNaive):
  test = test.copy() # Copying the test data

  plt.figure(figsize=(10,6)) # Setting the figure size

  test.index = test['Time'] # Setting the index to the 'Time' column

  # Plotting the graph
  plt.plot(test['Total_feeder'], label='Testing Data', color='orange') 
  plt.plot(test.index, predictionsNaive, label='Naive Forecast', color='red')

  # Adding the title, x-axis label, y-axis label and legend
  plt.title('Naive Model Forecast')
  plt.xlabel('Date')
  plt.ylabel('Total Feeder')
  plt.legend()

  plt.show() # Displaying the graph

# Plotting the seasonal naive model
def plot_seasonal_naive_model(train, test, predictionsSeasonalNaive):
  test = test.copy() # Copying the test data
   
  plt.figure(figsize=(10,6)) # Setting the figure size

  test.index = test['Time'] # Setting the index to the 'Time' column

  # Plotting the graph
  plt.plot(test['Total_feeder'], label = 'Testing Data', color = 'orange')
  plt.plot(test.index, predictionsSeasonalNaive, label = 'Seasonal Naive Forecast', color = 'red')
  
  # Adding the title, x-axis label, y-axis label and legend
  plt.title('Seasonal Naive Model Forecast')
  plt.xlabel('Date')
  plt.ylabel('Total Feeder')
  plt.legend()

  plt.show() # Displaying the graph

# Plotting the linear regression model
def plot_linear_regression_model(train, test, predictionsLinear):
  test = test.copy() # Copying the test data

  plt.figure(figsize=(10,6)) # Setting the figure size

  test.index = test['Time'] # Setting the index to the 'Time' column

  # Plotting the graph
  plt.plot(test['Total_feeder'], label='Testing Data', color='orange')
  plt.plot(test.index, predictionsLinear, label='Linear Regression Forecast', color='red')

  # Adding the title, x-axis label, y-axis label and legend
  plt.title('Linear Regression Model Forecast')
  plt.xlabel('Date')
  plt.ylabel('Total Feeder')
  plt.legend()

  plt.show() # Displaying the graph

def plot_ARIMA_model(train, test, test_forecast_series):

  # Set the index to the 'Time' column
  train = train.set_index('Time')
  test = test.set_index('Time')

  # Resample the data to hourly
  train = train.asfreq('h')
  test = test.asfreq('h')

  plt.figure(figsize=(10,6))  # Plot the ARIMA forecast

  # Plotting the graph
  plt.plot(test['Total_feeder'], label='Testing Data', color='orange')
  plt.plot(test_forecast_series, label='ARIMA Forecast', color='red') 

  # Adding the title, x-axis label, y-axis label and legend
  plt.title('ARIMA Model Forecast')
  plt.xlabel('Date')
  plt.ylabel('Total Feeder')
  plt.legend()

  plt.show() # Displaying the graph


### MAIN FUNCTION ###


def main(df):
  df = time_period(df) # Converting the string into a datetime format

  # Grouping the data by Weeks
  full_weeks = week_grouping(df)
  week_list = list(full_weeks.keys()) # Getting the list of weeks

  # Loop through the first 5 weeks
  for i, week_name in enumerate(week_list[:5]):  
        print(f"\nProcessing Week {i + 1}: {week_name}\n") # Print the current week being processed

        train, test = train_test_split(week_list, full_weeks, i) # Get train and test data for the current week

        best_order, best_rmse, best_forecast = evaluate_models(train, test) # # Evaluate best order for current ARIMA model

        best_forecast_series = pd.Series(best_forecast, index=test.set_index('Time').index) # Convert forecast to pandas Series

        # Run prediction models
        predictionsLinear, linearMSE = linear_regression(train, test)
        predictionsNaive, naiveMSE = naive_model(train, test)
        predictionsSeasonalNaive, seasonalMSE = seasonal_naive_model(train, test)
        ARIMAMSE, test_forecast_series = ARIMA_test(test, train, best_order)

        # Data analysis
        dataAnalysis(predictionsLinear, predictionsNaive, predictionsSeasonalNaive, linearMSE, naiveMSE, seasonalMSE, ARIMAMSE)

        # Plot predictions
        plot_naive_model(train, test, predictionsNaive)
        plot_seasonal_naive_model(train, test, predictionsSeasonalNaive)
        plot_linear_regression_model(train, test, predictionsLinear)
        plot_ARIMA_model(train, test, test_forecast_series)
  
main(df) # Calling the main function

  
