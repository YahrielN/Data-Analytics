import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the Dataset
# Assuming "marketing_campaign.csv" is placed in the project folder
df = pd.read_csv('test.csv')

# Preview the first few rows
print(df.head())

# Data Preprocessing
# Select relevant columns
df = df[['id', 'duration', 'age', 'numberOfContacts', 'daySinceLastCampaign', 'numberOfContactsLastCampaign']]

# Drop rows with missing values
df.dropna(inplace=True)

# Calculate new metrics: ROI, Conversion Rate, Cost per Conversion
df['ROI'] = (df['duration'] - df['age']) / df['age']
df['Conversion_Rate'] = df['numberOfContacts'] / df['daySinceLastCampaign']
df['Cost_Per_Conversion'] = df['duration'] / df['numberOfContactsLastCampaign']

# Preview the dataset after calculations
print(df.head())

# ROI Visualization
plt.figure(figsize=(10, 6))
plt.bar(df['id'], df['ROI'], color='blue')
plt.xlabel('Campaign ID')
plt.ylabel('ROI')
plt.title('ROI per Campaign')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Conversion Rate Visualization
fig = px.bar(df, x='id', y='Conversion_Rate', title='Conversion Rate per Campaign')
fig.show()

# Cost per Conversion Visualization
plt.figure(figsize=(10, 6))
plt.plot(df['id'], df['Cost_Per_Conversion'], marker='o', linestyle='-', color='green')
plt.xlabel('Campaign ID')
plt.ylabel('Cost per Conversion')
plt.title('Cost per Conversion per Campaign')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Predictive Model for ROI
# Define features and target variable
X = df[['duration', 'age', 'numberOfContacts', 'daySinceLastCampaign']]
y = df['ROI']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Visualize Predictions vs Actual ROI
results = pd.DataFrame({'Actual ROI': y_test, 'Predicted ROI': y_pred})
results.reset_index(drop=True, inplace=True)

# Plotting Actual vs Predicted ROI using Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(results.index, results['Actual ROI'], label='Actual ROI', marker='o')
plt.plot(results.index, results['Predicted ROI'], label='Predicted ROI', marker='x')
plt.xlabel('Index')
plt.ylabel('ROI')
plt.title('Actual vs Predicted ROI')
plt.legend()
plt.tight_layout()
plt.show()
