import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv"
file_path_or_url = input("Enter the local file path or press Enter to use the default URL: ")
if file_path_or_url:
    sales_data = pd.read_csv(file_path_or_url, parse_dates=['Month'], index_col='Month')
else:
    sales_data = pd.read_csv(url, parse_dates=['Month'], index_col='Month')

# Print the first few rows to understand the dataset
print(sales_data.head())

# Plot the data to visualize the sales trend
sales_data.plot()
print(sales_data)
plt.title("Monthly Car Sales Over Time")
plt.ylabel("Sales")
plt.xlabel("Date")
plt.show()


# Rename columns for easier reference
sales_data.columns = ['Sales']

# Handle any missing values
sales_data.fillna(method='ffill', inplace=True)

# Print summary statistics to understand the data better
print(sales_data.describe())

# Plot ACF and PACF to determine the parameters for ARIMA
plot_acf(sales_data['Sales'])
print('ACF plot displayed')
plt.show()


plot_pacf(sales_data['Sales'])
print('PACF plot displayed')
plt.show()


# Fit the ARIMA model
model = ARIMA(sales_data['Sales'], order=(2, 1, 2))
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Monthly forecast
monthly_forecast = model_fit.forecast(steps=12)
print("Monthly Forecast Values:", monthly_forecast.values)  # Forecasting for the next 12 months

# Plot the forecasted values
plt.plot(sales_data.index, sales_data['Sales'], label='Historical Sales')
print('Monthly sales data plotted')
plt.plot(pd.date_range(sales_data.index[-1], periods=12, freq='M'), monthly_forecast, color='red', label='Forecasted Sales')
plt.title("Monthly Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()


# Resample the data to weekly frequency
weekly_data = sales_data.resample('W').ffill()

# Fit a new ARIMA model for the weekly data
weekly_model = ARIMA(weekly_data['Sales'], order=(2, 1, 2))
weekly_model_fit = weekly_model.fit()

# Weekly forecast
weekly_forecast = weekly_model_fit.forecast(steps=12)
print("Weekly Forecast Values:", weekly_forecast.values)  # Forecasting for the next 12 weeks

# Plot the weekly forecast
plt.plot(weekly_data.index, weekly_data['Sales'], label='Historical Sales')
print('Weekly sales data plotted')
plt.plot(pd.date_range(weekly_data.index[-1], periods=12, freq='W'), weekly_forecast, color='red', label='Forecasted Sales')
plt.title("Weekly Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()


# Split data into training and testing sets
train_data = sales_data['Sales'][:-12]
test_data = sales_data['Sales'][-12:]

# Train the model on the training dataset
model = ARIMA(train_data, order=(2, 1, 2))
model_fit = model.fit()

# Predict on test dataset
predictions = model_fit.forecast(steps=12)
mse = mean_squared_error(test_data, predictions)

print(f"Mean Squared Error: {mse}")
